//! Minimal OpenAI-compatible HTTP server above [`vllm_chat`].

mod config;
mod error;
mod grpc;
mod listener;
mod lora;
mod middleware;
mod routes;
mod server_info;
mod state;
#[cfg(feature = "openssl")]
mod tls;
mod utils;

use std::sync::{Arc, OnceLock};

use anyhow::{Context as _, Result};
use axum::Router;
use axum::serve::ListenerExt as _;
pub use config::{
    ApiServerOptions, Config, CoordinatorMode, CorsConfig, HttpListenerMode, TlsConfig,
};
use tokio::net::TcpListener;
use tokio::time::{Instant, sleep_until};
use tokio_stream::wrappers::TcpListenerStream;
use tokio_util::either::Either;
use tokio_util::sync::CancellationToken;
use tonic::transport::Server as TonicServer;
use tracing::{info, trace, warn};
use vllm_chat::{ChatLlm, LoadModelBackendsOptions, load_model_backends};
pub use vllm_chat::{ChatTemplateContentFormatOption, ParserSelection, RendererSelection};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};
use vllm_llm::Llm;
use vllm_text::TextLlm;

use crate::listener::Listener;
use crate::routes::build_router;
use crate::server_info::ServerInfoSnapshot;
use crate::state::AppState;

/// Resolve the public model names accepted by the frontend.
fn effective_served_model_names(model: &str, served_model_name: &[String]) -> Vec<String> {
    if served_model_name.is_empty() {
        vec![model.to_string()]
    } else {
        served_model_name.to_vec()
    }
}

/// Build the shared application state for one configured model and one engine
/// client.
async fn build_state(config: &Config) -> Result<Arc<AppState>> {
    // If no served names are specified, fall back to the backend model path so
    // that the API always has at least one valid model ID. Use the same primary
    // public name for frontend-side metrics labels.
    let served_model_names = effective_served_model_names(&config.model, &config.served_model_name);
    let metrics_model_name = served_model_names[0].clone();

    // Load both backends from the same model metadata so they stay in sync.
    let loaded = load_model_backends(
        &config.model,
        LoadModelBackendsOptions {
            renderer: config.renderer,
            language_model_only: config.language_model_only,
            chat_template: config.chat_template.clone(),
            chat_template_content_format: config.chat_template_content_format,
            default_chat_template_kwargs: config
                .default_chat_template_kwargs
                .clone()
                .unwrap_or_default(),
        },
    )
    .await
    .context("failed to create chat/text backends")?;
    let text_backend = loaded.text_backend;
    let chat_backend = loaded.chat_backend;

    let coordinator_mode = config.effective_coordinator_mode(text_backend.is_moe());
    info!(
        engine_count = config.engine_count(),
        model_is_moe = text_backend.is_moe(),
        ?coordinator_mode,
        "resolved coordinator mode"
    );

    let client = EngineCoreClient::connect(EngineCoreClientConfig {
        transport_mode: config.transport_mode.clone(),
        coordinator_mode,
        model_name: metrics_model_name,
        client_index: 0,
    })
    .await
    .context("failed to connect to engine core")?;

    let llm = Llm::new(client).with_log_stats(!config.disable_log_stats);
    let text = TextLlm::new(llm, text_backend).with_max_logprobs(config.max_logprobs);

    let chat = ChatLlm::new(text, chat_backend)
        .with_tool_call_parser(config.tool_call_parser.clone())
        .with_reasoning_parser(config.reasoning_parser.clone());

    Ok(Arc::new(
        AppState::new(served_model_names, chat)
            .with_api_server_options(config.api_server_options)
            .with_server_info(ServerInfoSnapshot::from_config(config))
            .with_api_keys(config.api_keys.clone())
            .with_cors(config.cors.clone()),
    ))
}

/// Run the OpenAI-compatible HTTP server until the supplied shutdown token is
/// cancelled.
///
/// The server owns one `vllm-chat` facade, which in turn owns the lower
/// `vllm-text` and `vllm-llm` layers, and shuts them down before returning.
pub async fn serve(config: Config, shutdown: CancellationToken) -> Result<()> {
    serve_with_router_extension(config, shutdown, |router| router).await
}

/// Run the OpenAI-compatible HTTP server with an opt-in router extension.
///
/// The extension receives the finalized vLLM router and can merge additional
/// routes before the server starts accepting requests.
pub async fn serve_with_router_extension<F>(
    config: Config,
    shutdown: CancellationToken,
    extend_router: F,
) -> Result<()>
where
    F: FnOnce(Router) -> Router,
{
    config.validate().context("invalid OpenAI frontend configuration")?;

    // Also check shutdown during the (potentially long) startup handshake.
    let state = tokio::select! {
        result = build_state(&config) => result?,
        _ = shutdown.cancelled() => return Ok(()),
    };
    let listener = Listener::bind(&config.listener_mode)
        .await
        .context("failed to bind listener for OpenAI server")?;
    let bind_address = listener.local_addr()?;
    let model = state.primary_model_name().to_owned();
    let app = extend_router(build_router(state.clone()));

    // Optionally bind the gRPC Generate server on a separate port. Bind
    // synchronously here so bind errors (port in use, permission denied, ...)
    // surface before we start serving, rather than being deferred until
    // shutdown. The gRPC listener follows the same host as the HTTP listener so
    // that enabling --grpc-port does not accidentally expose the service on all
    // interfaces when HTTP is intentionally local-only.
    let grpc_setup = if let Some(grpc_port) = config.grpc_port {
        let grpc_host = match &config.listener_mode {
            HttpListenerMode::BindTcp { host, .. } => host.as_str(),
            HttpListenerMode::BindUnix { .. } | HttpListenerMode::InheritedFd { .. } => "0.0.0.0",
        };
        let grpc_listener = TcpListener::bind((grpc_host, grpc_port))
            .await
            .with_context(|| format!("failed to bind gRPC listener on {grpc_host}:{grpc_port}"))?;
        let addr = grpc_listener.local_addr()?;
        let svc = grpc::GenerateServer::new(grpc::GenerateServiceImpl::new(state.clone()));
        info!(%addr, "starting gRPC server");
        Some((grpc_listener, svc))
    } else {
        None
    };

    info!(%bind_address, %model, "starting OpenAI server");

    // Run HTTP and gRPC concurrently under a child token of the caller's shutdown
    // token. Caller cancellation propagates into both protocols; if either
    // protocol exits first, we cancel this child token so its sibling also
    // begins a graceful drain.
    let server_shutdown = shutdown.child_token();
    let force_shutdown = CancellationToken::new();
    let shutdown_deadline = Arc::new(OnceLock::new());

    // Spawn a task to trigger `force_shutdown` after shutdown deadline elapses.
    tokio::spawn({
        let shutdown = server_shutdown.clone();
        let force_shutdown = force_shutdown.clone();
        let shutdown_deadline = shutdown_deadline.clone();
        let shutdown_timeout = config.shutdown_timeout;

        async move {
            shutdown.cancelled().await;
            let deadline = Instant::now() + shutdown_timeout;
            let _ = shutdown_deadline.set(deadline);

            if shutdown_timeout.is_zero() {
                force_shutdown.cancel();
            } else {
                sleep_until(deadline).await;
                force_shutdown.cancel();
            }
        }
    });

    // Plaintext HTTP via `axum::serve`, or (openssl build) TLS termination via a
    // spawned-per-connection accept loop when `config.tls` is set.
    #[cfg(feature = "openssl")]
    let http_fut = match config.tls.clone() {
        Some(tls_cfg) => {
            let acceptor: Arc<dyn tls::TlsAcceptor> =
                tls::OpensslAcceptor::new(&tls_cfg).context("failed to build TLS acceptor")?;
            let tcp = match listener {
                Listener::Tcp(tcp) => tcp,
                Listener::Unix(_) => anyhow::bail!("TLS termination requires a TCP listener"),
            };
            if tls_cfg.enable_refresh {
                tls::spawn_refresher(acceptor.clone(), server_shutdown.clone());
            }
            let shutdown = server_shutdown.child_token();
            Either::Left(serve_http_tls(
                tcp,
                app,
                acceptor,
                shutdown,
                server_shutdown.clone(),
                force_shutdown.clone(),
            ))
        }
        None => {
            let shutdown = server_shutdown.child_token();
            Either::Right(serve_http_plain(
                listener,
                app,
                shutdown,
                server_shutdown.clone(),
                force_shutdown.clone(),
            ))
        }
    };
    #[cfg(not(feature = "openssl"))]
    let http_fut = {
        let shutdown = server_shutdown.child_token();
        serve_http_plain(
            listener,
            app,
            shutdown,
            server_shutdown.clone(),
            force_shutdown.clone(),
        )
    };

    let grpc_fut = {
        let shutdown = server_shutdown.child_token();
        let server_shutdown = server_shutdown.clone();
        let force_shutdown = force_shutdown.clone();
        async move {
            let Some((grpc_listener, svc)) = grpc_setup else {
                // No gRPC configured: just wait for shutdown so we do not race the
                // join! by resolving early and tripping the cancellation token.
                shutdown.cancelled().await;
                return Ok(());
            };
            let server = TonicServer::builder().add_service(svc).serve_with_incoming_shutdown(
                TcpListenerStream::new(grpc_listener),
                shutdown.cancelled_owned(),
            );

            let result = tokio::select! {
                result = server => {
                    result.context("gRPC server failed")
                }
                _ = force_shutdown.cancelled() => {
                    warn!("gRPC graceful shutdown deadline elapsed; aborting server");
                    Ok(())
                }
            };

            server_shutdown.cancel();
            result
        }
    };

    let (http_res, grpc_res) = tokio::join!(http_fut, grpc_fut);
    http_res.and(grpc_res)?;

    let shutdown_deadline = shutdown_deadline
        .get()
        .copied()
        .unwrap_or_else(|| Instant::now() + config.shutdown_timeout);
    state.shutdown(shutdown_deadline).await
}

/// Serve plaintext HTTP via `axum::serve`, draining gracefully on `shutdown` and
/// aborting if `force_shutdown` (the deadline) fires first.
async fn serve_http_plain(
    listener: Listener,
    app: Router,
    shutdown: CancellationToken,
    server_shutdown: CancellationToken,
    force_shutdown: CancellationToken,
) -> Result<()> {
    // TCP_NODELAY on every accepted connection; don't let Nagle sit on small replies.
    let listener = listener.tap_io(|io| {
        if let Either::Left(tcp_stream) = io
            && let Err(err) = tcp_stream.set_nodelay(true)
        {
            trace!(error = %err, "failed to enable TCP_NODELAY on accepted HTTP connection");
        }
    });

    let server = axum::serve(listener, app).with_graceful_shutdown(shutdown.cancelled_owned());
    let result = tokio::select! {
        result = server => result.context("HTTP server failed"),
        _ = force_shutdown.cancelled() => {
            warn!("HTTP graceful shutdown deadline elapsed; aborting server");
            Ok(())
        }
    };
    server_shutdown.cancel();
    result
}

/// Serve HTTPS by terminating TLS per connection on a spawned task (no
/// head-of-line blocking on slow handshakes), then driving the axum router over
/// the decrypted stream via hyper. Drains in-flight connections on `shutdown`
/// until `force_shutdown` elapses.
#[cfg(feature = "openssl")]
async fn serve_http_tls(
    tcp: TcpListener,
    app: Router,
    acceptor: Arc<dyn tls::TlsAcceptor>,
    shutdown: CancellationToken,
    server_shutdown: CancellationToken,
    force_shutdown: CancellationToken,
) -> Result<()> {
    use std::time::Duration;

    use hyper_util::rt::{TokioExecutor, TokioIo};
    use hyper_util::server::conn::auto::Builder;
    use hyper_util::service::TowerToHyperService;
    use tokio_util::task::TaskTracker;

    const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
    let conns = TaskTracker::new();

    loop {
        let (sock, peer) = tokio::select! {
            biased;
            _ = shutdown.cancelled() => break,
            accept = tcp.accept() => match accept {
                Ok(pair) => pair,
                Err(err) => {
                    warn!(%err, "failed to accept HTTPS connection");
                    continue;
                }
            },
        };
        if let Err(err) = sock.set_nodelay(true) {
            trace!(error = %err, "failed to enable TCP_NODELAY on accepted HTTPS connection");
        }

        let acceptor = acceptor.clone();
        let app = app.clone();
        let force_shutdown = force_shutdown.clone();
        conns.spawn(async move {
            let stream = match tokio::time::timeout(HANDSHAKE_TIMEOUT, acceptor.accept(sock)).await
            {
                Ok(Ok(stream)) => stream,
                Ok(Err(err)) => return log_tls_noise(peer, "TLS handshake error", &err),
                Err(_) => return tracing::debug!(%peer, "TLS handshake timed out"),
            };

            let service = TowerToHyperService::new(app);
            let builder = Builder::new(TokioExecutor::new());
            let conn = builder.serve_connection_with_upgrades(TokioIo::new(stream), service);
            tokio::pin!(conn);
            let result = tokio::select! {
                result = conn.as_mut() => result,
                _ = force_shutdown.cancelled() => return,
            };
            if let Err(err) = result {
                log_conn_noise(peer, &err);
            }
        });
    }

    // Stop accepting and let in-flight connections finish until the deadline.
    conns.close();
    tokio::select! {
        _ = conns.wait() => {}
        _ = force_shutdown.cancelled() => {
            warn!("HTTPS graceful shutdown deadline elapsed; aborting in-flight connections");
        }
    }
    server_shutdown.cancel();
    Ok(())
}

/// Demote expected handshake churn (health probes, client aborts, mTLS
/// rejections) to debug; anything else is a real warning.
#[cfg(feature = "openssl")]
fn log_tls_noise(peer: std::net::SocketAddr, context: &str, err: impl std::fmt::Display) {
    let msg = err.to_string().to_lowercase();
    if [
        "connection reset by peer",
        "unexpected eof",
        "protocol error",
        "handshake failure",
    ]
    .iter()
    .any(|needle| msg.contains(needle))
    {
        tracing::debug!(%peer, error = %err, "{context}");
    } else {
        warn!(%peer, error = %err, "{context}");
    }
}

/// Same idea for errors while serving an established connection.
#[cfg(feature = "openssl")]
fn log_conn_noise(peer: std::net::SocketAddr, err: impl std::fmt::Display) {
    let msg = err.to_string().to_lowercase();
    if [
        "connection reset by peer",
        "broken pipe",
        "connection was closed",
    ]
    .iter()
    .any(|needle| msg.contains(needle))
    {
        tracing::debug!(%peer, error = %err, "connection closed");
    } else {
        warn!(%peer, error = %err, "error serving HTTPS connection");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_served_model_names_falls_back_to_backend_model() {
        assert_eq!(
            effective_served_model_names("backend-model", &[]),
            vec!["backend-model"]
        );
    }

    #[test]
    fn effective_served_model_names_preserves_public_names() {
        let served_names = vec!["public-model".to_string(), "public-alias".to_string()];

        assert_eq!(
            effective_served_model_names("backend-model", &served_names),
            served_names
        );
    }

    /// Drive the real `serve_http_tls` accept loop with on-disk certs and a
    /// trivial router (no engine), fire an actual HTTPS request, and assert the
    /// response comes back over TLS. Covers the handshake, hyper-over-axum, and
    /// graceful shutdown.
    #[cfg(feature = "openssl")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn serves_https_through_the_tls_accept_loop() {
        use axum::routing::get;
        use tempfile::TempDir;

        use crate::tls::{OpensslAcceptor, TlsAcceptor, test_support};

        let dir = TempDir::new().unwrap();
        let (certfile, keyfile) = test_support::self_signed(&dir);
        let acceptor: Arc<dyn TlsAcceptor> =
            OpensslAcceptor::new(&test_support::tls_config(certfile.clone(), keyfile)).unwrap();

        let app = Router::new().route("/health", get(|| async { "ok" }));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let shutdown = CancellationToken::new();
        let server = tokio::spawn(serve_http_tls(
            listener,
            app,
            acceptor,
            shutdown.child_token(),
            shutdown.clone(),
            CancellationToken::new(),
        ));

        let response = test_support::https_get(addr, &certfile, "/health").await;
        assert!(
            response.starts_with("HTTP/1.1 200 OK"),
            "unexpected response: {response}"
        );
        assert!(response.ends_with("ok"), "unexpected body: {response}");

        shutdown.cancel();
        server.await.unwrap().unwrap();
    }
}
