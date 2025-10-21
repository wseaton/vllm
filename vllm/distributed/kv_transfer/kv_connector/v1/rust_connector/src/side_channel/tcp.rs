// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! simple TCP-based side channel implementation
//!
//! uses length-prefixed protocol:
//! - 4-byte big-endian length header
//! - followed by payload bytes

use super::error::{Result, SideChannelError};
use super::SideChannel;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// simple request message to request metadata
const REQUEST_MSG: &[u8] = b"GET_METADATA";

/// TCP-based side channel for metadata exchange
pub struct TcpSideChannel {
    listener_handle: Arc<Mutex<Option<thread::JoinHandle<()>>>>,
    running: Arc<AtomicBool>,
    metadata: Arc<Mutex<Option<Vec<u8>>>>,
}

impl TcpSideChannel {
    pub fn new() -> Self {
        Self {
            listener_handle: Arc::new(Mutex::new(None)),
            running: Arc::new(AtomicBool::new(false)),
            metadata: Arc::new(Mutex::new(None)),
        }
    }

    /// send length-prefixed message over stream
    fn send_message(stream: &mut TcpStream, data: &[u8]) -> Result<()> {
        let len = data.len() as u32;
        stream.write_all(&len.to_be_bytes())?;
        stream.write_all(data)?;
        stream.flush()?;
        Ok(())
    }

    /// receive length-prefixed message from stream
    fn recv_message(stream: &mut TcpStream, max_size: usize) -> Result<Vec<u8>> {
        let mut len_bytes = [0u8; 4];
        stream.read_exact(&mut len_bytes)?;
        let len = u32::from_be_bytes(len_bytes) as usize;

        if len > max_size {
            return Err(SideChannelError::Protocol(format!(
                "message too large: {} bytes (max {})",
                len, max_size
            )));
        }

        let mut buffer = vec![0u8; len];
        stream.read_exact(&mut buffer)?;
        Ok(buffer)
    }

    /// handle a single client connection
    fn handle_client(
        mut stream: TcpStream,
        metadata: &[u8],
    ) -> Result<()> {
        // receive request
        let request = Self::recv_message(&mut stream, REQUEST_MSG.len() + 1024)?;

        // validate request
        if request != REQUEST_MSG {
            return Err(SideChannelError::Protocol(format!(
                "unexpected request: {:?}",
                String::from_utf8_lossy(&request)
            )));
        }

        // send metadata response
        Self::send_message(&mut stream, metadata)?;

        Ok(())
    }

    /// listener thread function
    fn listener_thread(
        listener: TcpListener,
        metadata: Arc<Mutex<Option<Vec<u8>>>>,
        running: Arc<AtomicBool>,
    ) {
        // set non-blocking to allow periodic shutdown checks
        listener
            .set_nonblocking(true)
            .expect("failed to set non-blocking");

        while running.load(Ordering::SeqCst) {
            match listener.accept() {
                Ok((stream, _addr)) => {
                    let meta_guard = metadata.lock().unwrap();
                    if let Some(ref meta_bytes) = *meta_guard {
                        if let Err(e) = Self::handle_client(stream, meta_bytes) {
                            eprintln!("error handling client: {}", e);
                        }
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // no connections available, sleep briefly
                    thread::sleep(Duration::from_millis(50));
                }
                Err(e) => {
                    eprintln!("listener accept error: {}", e);
                    break;
                }
            }
        }
    }
}

impl Default for TcpSideChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl SideChannel for TcpSideChannel {
    fn start_listener(&mut self, port: u16, metadata: Vec<u8>) -> Result<()> {
        if self.running.load(Ordering::SeqCst) {
            return Err(SideChannelError::ServerAlreadyRunning { port });
        }

        // bind to all interfaces on specified port
        let addr = format!("0.0.0.0:{}", port);
        let listener = TcpListener::bind(&addr).map_err(|e| {
            SideChannelError::Protocol(format!("failed to bind to {}: {}", addr, e))
        })?;

        // store metadata
        *self.metadata.lock().unwrap() = Some(metadata);

        // start listener thread
        let running = Arc::clone(&self.running);
        let metadata = Arc::clone(&self.metadata);
        running.store(true, Ordering::SeqCst);

        let handle = thread::spawn(move || {
            Self::listener_thread(listener, metadata, running);
        });

        *self.listener_handle.lock().unwrap() = Some(handle);

        Ok(())
    }

    fn request_metadata(&self, host: &str, port: u16, timeout_ms: u64) -> Result<Vec<u8>> {
        let addr = format!("{}:{}", host, port);

        // resolve address
        let socket_addr = addr.to_socket_addrs()
            .map_err(|e| SideChannelError::AddressParse(format!("{}: {}", addr, e)))?
            .next()
            .ok_or_else(|| SideChannelError::AddressParse(format!("no addresses for {}", addr)))?;

        // connect with timeout
        let mut stream = TcpStream::connect_timeout(
            &socket_addr,
            Duration::from_millis(timeout_ms),
        )?;

        // set read/write timeouts
        stream.set_read_timeout(Some(Duration::from_millis(timeout_ms)))?;
        stream.set_write_timeout(Some(Duration::from_millis(timeout_ms)))?;

        // send request
        Self::send_message(&mut stream, REQUEST_MSG)?;

        // receive metadata (allow up to 10MB)
        let metadata = Self::recv_message(&mut stream, 10 * 1024 * 1024)?;

        Ok(metadata)
    }

    fn shutdown(&mut self) -> Result<()> {
        if !self.running.load(Ordering::SeqCst) {
            return Ok(());
        }

        // signal shutdown
        self.running.store(false, Ordering::SeqCst);

        // wait for listener thread to finish
        if let Some(handle) = self.listener_handle.lock().unwrap().take() {
            let _ = handle.join();
        }

        // clear metadata
        *self.metadata.lock().unwrap() = None;

        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tcp_side_channel_basic() {
        let mut server = TcpSideChannel::new();
        let client = TcpSideChannel::new();

        let test_metadata = b"test_metadata_bytes".to_vec();
        let port = 19876; // use non-standard port for testing

        // start server
        server
            .start_listener(port, test_metadata.clone())
            .expect("failed to start listener");
        assert!(server.is_running());

        // give server time to start
        thread::sleep(Duration::from_millis(100));

        // request metadata
        let received = client
            .request_metadata("127.0.0.1", port, 5000)
            .expect("failed to request metadata");

        assert_eq!(received, test_metadata);

        // shutdown
        server.shutdown().expect("failed to shutdown");
        assert!(!server.is_running());
    }

    #[test]
    fn test_tcp_side_channel_timeout() {
        let client = TcpSideChannel::new();

        // try to connect to non-existent server with short timeout
        let result = client.request_metadata("127.0.0.1", 19999, 100);
        assert!(result.is_err());
    }
}
