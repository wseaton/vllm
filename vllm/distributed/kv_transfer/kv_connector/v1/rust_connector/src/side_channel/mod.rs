// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! side channel for NIXL agent metadata exchange
//!
//! provides trait-based abstraction for different transport implementations
//! (TCP, ZMQ, HTTP, etc.) used to exchange agent metadata during handshake

pub mod error;
pub mod tcp;

use error::Result;

/// trait for side channel implementations used for NIXL agent handshake
///
/// implementations handle metadata exchange between vLLM deployments
pub trait SideChannel: Send + Sync {
    /// start listening for metadata requests on specified port
    ///
    /// # arguments
    /// * `port` - port number to listen on
    /// * `metadata` - agent metadata bytes to serve
    ///
    /// # returns
    /// ok if listener started successfully, error otherwise
    fn start_listener(&mut self, port: u16, metadata: Vec<u8>) -> Result<()>;

    /// request metadata from remote agent
    ///
    /// # arguments
    /// * `host` - hostname or IP address
    /// * `port` - port number
    /// * `timeout_ms` - timeout in milliseconds
    ///
    /// # returns
    /// metadata bytes from remote agent, or error if request fails
    fn request_metadata(&self, host: &str, port: u16, timeout_ms: u64) -> Result<Vec<u8>>;

    /// shutdown the side channel, stopping any listeners
    fn shutdown(&mut self) -> Result<()>;

    /// check if listener is currently running
    fn is_running(&self) -> bool;
}
