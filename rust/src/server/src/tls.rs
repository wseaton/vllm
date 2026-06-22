//! Inbound TLS termination: the backend-agnostic [`TlsAcceptor`] trait and its
//! system-OpenSSL implementation.

use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Context as _, Result};
use arc_swap::ArcSwap;
use openssl::ssl::{Ssl, SslAcceptor, SslFiletype, SslMethod, SslVerifyMode};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpStream;
use tokio_openssl::SslStream;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

use crate::config::TlsConfig;

pub(crate) trait TlsIo: AsyncRead + AsyncWrite + Send {}
impl<T: AsyncRead + AsyncWrite + Send> TlsIo for T {}

/// A decrypted byte stream from a terminated TLS connection,
/// using dyn for the type erasure.
pub(crate) type TlsStream = Pin<Box<dyn TlsIo>>;

/// dyn Trait used to in the future support different TLS backends
pub(crate) trait TlsAcceptor: Send + Sync + 'static {
    /// Run the server-side handshake on an accepted TCP connection.
    fn accept(&self, tcp: TcpStream) -> Pin<Box<dyn Future<Output = Result<TlsStream>> + Send>>;

    /// Rebuild from the original config and hot-swap
    fn reload(&self) -> Result<()>;

    /// Key/cert/CA paths to watch for rotation.
    fn watched_paths(&self) -> Vec<String>;
}

fn build_acceptor(cfg: &TlsConfig) -> Result<SslAcceptor> {
    let mut builder = SslAcceptor::mozilla_intermediate_v5(SslMethod::tls())
        .context("failed to create OpenSSL acceptor builder")?;
    builder
        .set_private_key_file(&cfg.keyfile, SslFiletype::PEM)
        .with_context(|| format!("failed to load ssl_keyfile {}", cfg.keyfile))?;
    builder
        .set_certificate_chain_file(&cfg.certfile)
        .with_context(|| format!("failed to load ssl_certfile {}", cfg.certfile))?;
    builder.check_private_key().context("ssl_keyfile does not match ssl_certfile")?;
    if let Some(ca) = &cfg.ca_certs {
        builder
            .set_ca_file(ca)
            .with_context(|| format!("failed to load ssl_ca_certs {ca}"))?;
    }
    // mirror python's stdlib ssl.CERT_NONE / CERT_OPTIONAL / CERT_REQUIRED.
    let verify = match cfg.cert_reqs {
        0 => SslVerifyMode::NONE,
        1 => SslVerifyMode::PEER,
        _ => SslVerifyMode::PEER | SslVerifyMode::FAIL_IF_NO_PEER_CERT,
    };
    builder.set_verify(verify);
    // OpenSSL cipher list governs TLS <=1.2, same as uvicorn's set_ciphers.
    if let Some(ciphers) = &cfg.ciphers {
        builder
            .set_cipher_list(ciphers)
            .with_context(|| format!("invalid ssl_ciphers {ciphers}"))?;
    }
    Ok(builder.build())
}

pub(crate) struct OpensslAcceptor {
    cfg: TlsConfig,
    acceptor: ArcSwap<SslAcceptor>,
}

impl OpensslAcceptor {
    pub(crate) fn new(cfg: &TlsConfig) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            cfg: cfg.clone(),
            acceptor: ArcSwap::from_pointee(build_acceptor(cfg)?),
        }))
    }
}

impl TlsAcceptor for OpensslAcceptor {
    fn accept(&self, tcp: TcpStream) -> Pin<Box<dyn Future<Output = Result<TlsStream>> + Send>> {
        let acceptor = self.acceptor.load_full();
        Box::pin(async move {
            let ssl = Ssl::new(acceptor.context()).context("failed to create SSL session")?;
            let mut stream = SslStream::new(ssl, tcp).context("failed to create SSL stream")?;
            Pin::new(&mut stream).accept().await.context("TLS handshake failed")?;
            Ok(Box::pin(stream) as TlsStream)
        })
    }

    fn reload(&self) -> Result<()> {
        self.acceptor.store(Arc::new(build_acceptor(&self.cfg)?));
        Ok(())
    }

    fn watched_paths(&self) -> Vec<String> {
        let mut paths = vec![self.cfg.keyfile.clone(), self.cfg.certfile.clone()];
        paths.extend(self.cfg.ca_certs.clone());
        paths
    }
}

/// Watch the acceptor's key/cert/CA files and reload it on change. Mirrors
/// Python's `SSLCertRefresher`: any change triggers a full reload; reload errors
/// are logged and watching continues. Runs until `shutdown` is cancelled.
pub(crate) fn spawn_refresher(acceptor: Arc<dyn TlsAcceptor>, shutdown: CancellationToken) {
    tokio::spawn(async move {
        use notify::{RecursiveMode, Watcher};

        let (tx, mut rx) = tokio::sync::mpsc::channel::<()>(8);
        let mut watcher =
            match notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
                if res.is_ok() {
                    // notify runs this on its own thread, so blocking_send is safe.
                    let _ = tx.blocking_send(());
                }
            }) {
                Ok(watcher) => watcher,
                Err(err) => {
                    error!(%err, "failed to start TLS cert watcher; rotation disabled");
                    return;
                }
            };

        let watched = acceptor.watched_paths();
        for path in &watched {
            if let Err(err) = watcher.watch(Path::new(path), RecursiveMode::NonRecursive) {
                error!(%err, path, "failed to watch TLS file; rotation may miss changes");
            }
        }
        info!(?watched, "watching TLS certificate files for rotation");

        loop {
            tokio::select! {
                _ = shutdown.cancelled() => break,
                msg = rx.recv() => {
                    if msg.is_none() {
                        break;
                    }
                    // Collapse a burst of events into one reload.
                    while rx.try_recv().is_ok() {}
                    match acceptor.reload() {
                        Ok(()) => info!("reloaded TLS certificates"),
                        Err(err) => warn!(%err, "failed to reload TLS certificates; keeping current"),
                    }
                }
            }
        }
    });
}

/// Shared TLS test helpers, used by this module's unit tests and by the
/// `serve_http_tls` integration test in `lib.rs`.
#[cfg(test)]
pub(crate) mod test_support {
    use std::io::Write;
    use std::net::SocketAddr;
    use std::pin::Pin;

    use openssl::asn1::Asn1Time;
    use openssl::hash::MessageDigest;
    use openssl::pkey::PKey;
    use openssl::rsa::Rsa;
    use openssl::ssl::{SslConnector, SslMethod};
    use openssl::x509::extension::SubjectAlternativeName;
    use openssl::x509::{X509, X509NameBuilder};
    use tempfile::TempDir;
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

    use super::TlsConfig;

    /// Write a fresh self-signed cert/key pair into `dir`, returning their paths.
    /// The cert carries a `localhost` SAN so clients can verify the hostname.
    pub(crate) fn self_signed(dir: &TempDir) -> (String, String) {
        let key = PKey::from_rsa(Rsa::generate(2048).unwrap()).unwrap();
        let mut name = X509NameBuilder::new().unwrap();
        name.append_entry_by_text("CN", "localhost").unwrap();
        let name = name.build();
        let mut builder = X509::builder().unwrap();
        builder.set_version(2).unwrap();
        builder.set_subject_name(&name).unwrap();
        builder.set_issuer_name(&name).unwrap();
        builder.set_pubkey(&key).unwrap();
        builder.set_not_before(&Asn1Time::days_from_now(0).unwrap()).unwrap();
        builder.set_not_after(&Asn1Time::days_from_now(1).unwrap()).unwrap();
        let san = SubjectAlternativeName::new()
            .dns("localhost")
            .build(&builder.x509v3_context(None, None))
            .unwrap();
        builder.append_extension(san).unwrap();
        builder.sign(&key, MessageDigest::sha256()).unwrap();
        let cert = builder.build();

        let cert_path = dir.path().join("tls.crt");
        let key_path = dir.path().join("tls.key");
        std::fs::File::create(&cert_path)
            .unwrap()
            .write_all(&cert.to_pem().unwrap())
            .unwrap();
        std::fs::File::create(&key_path)
            .unwrap()
            .write_all(&key.private_key_to_pem_pkcs8().unwrap())
            .unwrap();
        (
            cert_path.to_str().unwrap().to_string(),
            key_path.to_str().unwrap().to_string(),
        )
    }

    /// A minimal server TLS config pointing at the given cert/key (no client auth).
    pub(crate) fn tls_config(certfile: String, keyfile: String) -> TlsConfig {
        TlsConfig {
            keyfile,
            certfile,
            ca_certs: None,
            cert_reqs: 0,
            ciphers: None,
            enable_refresh: false,
        }
    }

    /// Issue a HTTPS GET over OpenSSL, trusting `ca_path` and verifying the
    /// `localhost` SAN, and return the raw HTTP response text. Uses
    /// `Connection: close` so the read terminates when the server closes.
    pub(crate) async fn https_get(addr: SocketAddr, ca_path: &str, path: &str) -> String {
        let mut connector = SslConnector::builder(SslMethod::tls_client()).unwrap();
        connector.set_ca_file(ca_path).unwrap();
        let ssl = connector.build().configure().unwrap().into_ssl("localhost").unwrap();
        let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
        let mut stream = tokio_openssl::SslStream::new(ssl, tcp).unwrap();
        Pin::new(&mut stream).connect().await.unwrap();

        let request =
            format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
        stream.write_all(request.as_bytes()).await.unwrap();
        stream.flush().await.unwrap();

        let mut response = Vec::new();
        let mut buf = [0u8; 4096];
        loop {
            // A clean HTTP close may surface as an OpenSSL EOF without close_notify;
            // either way we already have the full response, so stop reading.
            match stream.read(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(n) => response.extend_from_slice(&buf[..n]),
            }
        }
        String::from_utf8_lossy(&response).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::build_acceptor;
    use super::test_support::{self_signed, tls_config};

    #[test]
    fn builds_acceptor_from_self_signed_pem() {
        let dir = TempDir::new().unwrap();
        let (certfile, keyfile) = self_signed(&dir);
        assert!(build_acceptor(&tls_config(certfile, keyfile)).is_ok());
    }

    #[test]
    fn mtls_requires_ca_and_accepts_cipher_list() {
        let dir = TempDir::new().unwrap();
        let (certfile, keyfile) = self_signed(&dir);
        let mut cfg = tls_config(certfile.clone(), keyfile);
        // Verify CERT_REQUIRED with a CA bundle and an explicit TLS<=1.2 cipher list.
        cfg.ca_certs = Some(certfile);
        cfg.cert_reqs = 2;
        cfg.ciphers = Some("ECDHE-RSA-AES256-GCM-SHA384".to_string());
        assert!(build_acceptor(&cfg).is_ok());
    }

    #[test]
    fn rejects_key_cert_mismatch() {
        let dir = TempDir::new().unwrap();
        let (certfile, _) = self_signed(&dir);
        // A second, unrelated key that does not match the first cert.
        let other = TempDir::new().unwrap();
        let (_, other_key) = self_signed(&other);
        assert!(build_acceptor(&tls_config(certfile, other_key)).is_err());
    }
}
