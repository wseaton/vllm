// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Scheduler-side KV Connector Logic
//!
//! This module implements the scheduler-side state tracking and metadata
//! building for the NIXL KV connector. It tracks which requests need to
//! load KV from remote, which are ready to send, and builds metadata
//! for the worker-side to execute transfers.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

/// Request metadata for tracking remote KV transfers
#[pyclass]
#[derive(Debug, Clone)]
pub struct RequestMeta {
    /// Local block IDs allocated for this request
    #[pyo3(get)]
    pub local_block_ids: Vec<u32>,

    /// Remote block IDs to load from
    #[pyo3(get)]
    pub remote_block_ids: Vec<u32>,

    /// Remote engine ID (e.g., "prefill_engine_0")
    #[pyo3(get)]
    pub remote_engine_id: String,

    /// Remote host address
    #[pyo3(get)]
    pub remote_host: String,

    /// Remote port for side channel
    #[pyo3(get)]
    pub remote_port: u16,

    /// TP size of remote instance
    #[pyo3(get)]
    pub tp_size: u32,
}

#[pymethods]
impl RequestMeta {
    #[new]
    #[pyo3(signature = (local_block_ids, remote_block_ids, remote_engine_id, remote_host, remote_port, tp_size=1))]
    fn new(
        local_block_ids: Vec<u32>,
        remote_block_ids: Vec<u32>,
        remote_engine_id: String,
        remote_host: String,
        remote_port: u16,
        tp_size: u32,
    ) -> Self {
        Self {
            local_block_ids,
            remote_block_ids,
            remote_engine_id,
            remote_host,
            remote_port,
            tp_size,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RequestMeta(local_blocks={}, remote_blocks={}, engine={}, host={}:{})",
            self.local_block_ids.len(),
            self.remote_block_ids.len(),
            self.remote_engine_id,
            self.remote_host,
            self.remote_port
        )
    }
}

/// Scheduler-side connector for tracking request state
#[pyclass]
pub struct ConnectorScheduler {
    /// Engine ID for this instance
    engine_id: String,

    /// Block size (number of tokens per block)
    block_size: usize,

    /// Side channel host for metadata exchange
    side_channel_host: String,

    /// Side channel port (base port + DP rank * TP size)
    side_channel_port: u16,

    /// Requests that need to receive KV from remote
    /// Map: request_id -> (RequestMeta, num_external_tokens)
    reqs_need_recv: HashMap<String, (RequestMeta, u32)>,

    /// Requests that need to send KV to remote (with expiration time)
    /// Map: request_id -> expiration_timestamp
    reqs_need_send: HashMap<String, f64>,

    /// Requests currently in batch (for tracking)
    reqs_in_batch: HashSet<String>,

    /// Requests to remove from processed set (aborted, etc.)
    reqs_not_processed: HashSet<String>,
}

#[pymethods]
impl ConnectorScheduler {
    #[new]
    #[pyo3(signature = (engine_id, block_size, side_channel_host, side_channel_port))]
    fn new(
        engine_id: String,
        block_size: usize,
        side_channel_host: String,
        side_channel_port: u16,
    ) -> Self {
        Self {
            engine_id,
            block_size,
            side_channel_host,
            side_channel_port,
            reqs_need_recv: HashMap::new(),
            reqs_need_send: HashMap::new(),
            reqs_in_batch: HashSet::new(),
            reqs_not_processed: HashSet::new(),
        }
    }

    /// Get number of new tokens that can be loaded from external KV cache
    ///
    /// Returns:
    ///     (num_external_tokens, is_async): tuple of token count and whether
    ///     loading will be async (between scheduler steps)
    #[pyo3(signature = (request_id, num_computed_tokens, num_prompt_tokens, kv_transfer_params))]
    fn get_num_new_matched_tokens(
        &self,
        request_id: String,
        num_computed_tokens: u32,
        num_prompt_tokens: u32,
        kv_transfer_params: &Bound<'_, PyDict>,
    ) -> PyResult<(u32, bool)> {
        // Check if this is a remote prefill request
        if let Ok(Some(do_remote_prefill)) = kv_transfer_params.get_item("do_remote_prefill") {
            let do_remote_prefill: bool = do_remote_prefill.extract()?;

            if do_remote_prefill {
                // Remote prefill: get all prompt blocks from remote
                let count = num_prompt_tokens.saturating_sub(num_computed_tokens);
                if count > 0 {
                    return Ok((count, true));  // Async loading
                }
            }
        }

        // No remote prefill for this request
        Ok((0, false))
    }

    /// Update state after block allocation
    ///
    /// Called when the scheduler allocates blocks for a request that may
    /// load from or save to remote KV cache.
    #[pyo3(signature = (request_id, block_ids, num_external_tokens, kv_transfer_params))]
    fn update_state_after_alloc(
        &mut self,
        request_id: String,
        block_ids: Vec<u32>,
        num_external_tokens: u32,
        kv_transfer_params: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        // Check for remote decode (needs to send after completion)
        if let Ok(Some(do_remote_decode)) = kv_transfer_params.get_item("do_remote_decode") {
            let do_remote_decode: bool = do_remote_decode.extract()?;
            if do_remote_decode {
                self.reqs_in_batch.insert(request_id.clone());
            }
        }

        // Check for remote prefill (needs to receive)
        if let Ok(Some(do_remote_prefill)) = kv_transfer_params.get_item("do_remote_prefill") {
            let do_remote_prefill: bool = do_remote_prefill.extract()?;

            if do_remote_prefill {
                // Extract remote block IDs
                if let Ok(Some(remote_block_ids_obj)) = kv_transfer_params.get_item("remote_block_ids") {
                    let remote_block_ids: Vec<u32> = remote_block_ids_obj.extract()?;

                    if !remote_block_ids.is_empty() {
                        // Validate we have all required params
                        let remote_engine_id: String = kv_transfer_params
                            .get_item("remote_engine_id")?
                            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "Missing remote_engine_id in kv_transfer_params"
                            ))?
                            .extract()?;

                        let remote_host: String = kv_transfer_params
                            .get_item("remote_host")?
                            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "Missing remote_host in kv_transfer_params"
                            ))?
                            .extract()?;

                        let remote_port: u16 = kv_transfer_params
                            .get_item("remote_port")?
                            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "Missing remote_port in kv_transfer_params"
                            ))?
                            .extract()?;

                        let tp_size: u32 = kv_transfer_params
                            .get_item("tp_size")
                            .ok()
                            .and_then(|opt| opt)
                            .map(|v| v.extract().unwrap_or(1))
                            .unwrap_or(1);

                        // Determine local block IDs to use
                        // If num_external_tokens == 0, we have full prefix cache hit
                        let local_block_ids = if num_external_tokens > 0 {
                            block_ids
                        } else {
                            Vec::new()  // No blocks needed for full cache hit
                        };

                        let meta = RequestMeta {
                            local_block_ids,
                            remote_block_ids,
                            remote_engine_id,
                            remote_host,
                            remote_port,
                            tp_size,
                        };

                        self.reqs_need_recv.insert(request_id.clone(), (meta, num_external_tokens));
                    }
                }

                // Mark do_remote_prefill as False to prevent duplicate processing
                // (Python side should do this, but we note it here)
            }
        }

        Ok(())
    }

    /// Build connector metadata for worker-side
    ///
    /// Returns a dictionary with request metadata for loading/sending KV.
    /// This resets the scheduler state for the next step.
    fn build_connector_meta<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let meta = PyDict::new_bound(py);

        // Build reqs_to_recv dict
        let reqs_to_recv = PyDict::new_bound(py);
        for (req_id, (req_meta, _num_external_tokens)) in self.reqs_need_recv.drain() {
            // Convert RequestMeta to dict
            let meta_dict = PyDict::new_bound(py);
            meta_dict.set_item("local_block_ids", req_meta.local_block_ids)?;
            meta_dict.set_item("remote_block_ids", req_meta.remote_block_ids)?;
            meta_dict.set_item("remote_engine_id", req_meta.remote_engine_id)?;
            meta_dict.set_item("remote_host", req_meta.remote_host)?;
            meta_dict.set_item("remote_port", req_meta.remote_port)?;
            meta_dict.set_item("tp_size", req_meta.tp_size)?;

            reqs_to_recv.set_item(req_id, meta_dict)?;
        }
        meta.set_item("reqs_to_recv", reqs_to_recv)?;

        // Build reqs_to_send dict
        let reqs_to_send = PyDict::new_bound(py);
        for (req_id, expiration_time) in &self.reqs_need_send {
            reqs_to_send.set_item(req_id, expiration_time)?;
        }
        meta.set_item("reqs_to_send", reqs_to_send)?;

        // Build reqs_in_batch set
        let reqs_in_batch = PyList::new_bound(py, &self.reqs_in_batch);
        meta.set_item("reqs_in_batch", reqs_in_batch)?;

        // Build reqs_not_processed set
        let reqs_not_processed = PyList::new_bound(py, &self.reqs_not_processed);
        meta.set_item("reqs_not_processed", reqs_not_processed)?;

        // Clear state for next step
        self.reqs_in_batch.clear();
        self.reqs_not_processed.clear();
        self.reqs_need_send.clear();

        Ok(meta)
    }

    /// Called when a request finishes
    ///
    /// Returns:
    ///     (delay_free_blocks, kv_transfer_params): tuple indicating whether
    ///     to delay freeing blocks and optional params for remote consumption
    #[pyo3(signature = (request_id, block_ids, request_status, kv_transfer_params, abort_request_timeout_sec))]
    fn request_finished<'py>(
        &mut self,
        py: Python<'py>,
        request_id: String,
        block_ids: Vec<u32>,
        request_status: String,
        kv_transfer_params: Option<&Bound<'py, PyDict>>,
        abort_request_timeout_sec: f64,
    ) -> PyResult<(bool, Option<Bound<'py, PyDict>>)> {
        let Some(params) = kv_transfer_params else {
            // No KV transfer params, free blocks immediately
            return Ok((false, None));
        };

        // Check if remote prefill flag is still set (request aborted before scheduling)
        if let Ok(Some(do_remote_prefill)) = params.get_item("do_remote_prefill") {
            let do_remote_prefill: bool = do_remote_prefill.extract()?;
            if do_remote_prefill {
                // Add to reqs_need_recv with empty blocks so we notify prefill instance
                let remote_engine_id: String = params
                    .get_item("remote_engine_id")?
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Missing remote_engine_id"
                    ))?
                    .extract()?;

                let remote_host: String = params
                    .get_item("remote_host")?
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Missing remote_host"
                    ))?
                    .extract()?;

                let remote_port: u16 = params
                    .get_item("remote_port")?
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Missing remote_port"
                    ))?
                    .extract()?;

                let meta = RequestMeta {
                    local_block_ids: Vec::new(),
                    remote_block_ids: Vec::new(),
                    remote_engine_id,
                    remote_host,
                    remote_port,
                    tp_size: 1,
                };

                self.reqs_need_recv.insert(request_id, (meta, 0));
                return Ok((false, None));
            }
        }

        // Check for remote decode (sending to remote)
        if let Ok(Some(do_remote_decode)) = params.get_item("do_remote_decode") {
            let do_remote_decode: bool = do_remote_decode.extract()?;
            if !do_remote_decode {
                return Ok((false, None));
            }
        } else {
            return Ok((false, None));
        }

        // Check if request finished with length cap (needs to be sent)
        if request_status != "FINISHED_LENGTH_CAPPED" {
            // Aborted or other status - don't send
            self.reqs_not_processed.insert(request_id);
            return Ok((false, None));
        }

        // Delay freeing blocks if we have any
        let delay_free_blocks = !block_ids.is_empty();

        if delay_free_blocks {
            // Calculate expiration time
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();
            let expiration = now + abort_request_timeout_sec;

            self.reqs_need_send.insert(request_id, expiration);

            // Build kv_transfer_params for decode instance
            let new_params = PyDict::new_bound(py);
            new_params.set_item("do_remote_prefill", true)?;
            new_params.set_item("do_remote_decode", false)?;
            new_params.set_item("remote_block_ids", block_ids)?;
            new_params.set_item("remote_engine_id", &self.engine_id)?;
            new_params.set_item("remote_host", &self.side_channel_host)?;
            new_params.set_item("remote_port", self.side_channel_port)?;
            // TODO: get TP size from config
            new_params.set_item("tp_size", 1)?;

            Ok((true, Some(new_params)))
        } else {
            Ok((false, None))
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ConnectorScheduler(engine={}, reqs_recv={}, reqs_send={}, in_batch={})",
            self.engine_id,
            self.reqs_need_recv.len(),
            self.reqs_need_send.len(),
            self.reqs_in_batch.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_meta_creation() {
        let meta = RequestMeta::new(
            vec![0, 1, 2],
            vec![10, 11, 12],
            "test_engine".to_string(),
            "localhost".to_string(),
            8000,
            1,
        );

        assert_eq!(meta.local_block_ids, vec![0, 1, 2]);
        assert_eq!(meta.remote_block_ids, vec![10, 11, 12]);
        assert_eq!(meta.remote_engine_id, "test_engine");
        assert_eq!(meta.remote_host, "localhost");
        assert_eq!(meta.remote_port, 8000);
        assert_eq!(meta.tp_size, 1);
    }

    #[test]
    fn test_scheduler_creation() {
        let sched = ConnectorScheduler::new(
            "test_engine".to_string(),
            16,
            "0.0.0.0".to_string(),
            8000,
        );

        assert_eq!(sched.engine_id, "test_engine");
        assert_eq!(sched.block_size, 16);
        assert!(sched.reqs_need_recv.is_empty());
        assert!(sched.reqs_need_send.is_empty());
    }
}
