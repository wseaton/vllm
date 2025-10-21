// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! error types for side channel operations

use std::io;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SideChannelError {
    #[error("network i/o error: {0}")]
    Io(#[from] io::Error),

    #[error("connection timeout after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    #[error("protocol error: {0}")]
    Protocol(String),

    #[error("invalid metadata: {0}")]
    InvalidMetadata(String),

    #[error("server already running on port {port}")]
    ServerAlreadyRunning { port: u16 },

    #[error("server not running")]
    ServerNotRunning,

    #[error("failed to parse address: {0}")]
    AddressParse(String),
}

pub type Result<T> = std::result::Result<T, SideChannelError>;
