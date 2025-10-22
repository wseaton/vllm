// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Worker-side KV Connector Logic
//!
//! This module implements the worker-side KV cache registration and transfer
//! operations. It handles NIXL memory registration, handshakes with remote
//! instances, initiating async transfers, and polling for completion.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet, IntoPyDict};
#[cfg(feature = "nixl")]
use pyo3::types::PyList;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

/// Information about a registered KV cache layer
#[derive(Debug, Clone)]
struct KVCacheInfo {
    /// Base memory address of the KV cache
    base_addr: usize,
    /// Total size in bytes
    size: usize,
    /// Number of blocks in this cache
    num_blocks: usize,
    /// Length of each block in bytes
    block_len: usize,
    /// Device ID (GPU index)
    dev_id: u64,
}

/// Metadata for a transfer request
#[derive(Debug, Clone)]
struct TransferMeta {
    /// Request ID
    request_id: String,
    /// Local block IDs to write to
    local_block_ids: Vec<u32>,
    /// Remote block IDs to read from
    remote_block_ids: Vec<u32>,
    /// Remote engine ID
    remote_engine_id: String,
    /// Remote host
    remote_host: String,
    /// Remote port
    remote_port: u16,
}

/// Worker-side connector for executing KV transfers
#[pyclass]
pub struct ConnectorWorker {
    /// Engine ID for this instance
    engine_id: String,

    /// Tensor parallelism rank
    tp_rank: u64,

    /// Block size (number of tokens per block)
    block_size: usize,

    /// KV cache information by layer name
    /// Map: layer_name -> KVCacheInfo
    kv_caches: Arc<Mutex<HashMap<String, KVCacheInfo>>>,

    /// NIXL agent for this worker (optional, only when NIXL feature is enabled)
    /// Stored as a Py<NixlAgent> reference to use Rust bindings
    #[cfg(feature = "nixl")]
    nixl_agent: Option<Py<crate::NixlAgent>>,

    /// Local transfer descriptor handle (for source operations)
    src_xfer_handle: Arc<Mutex<Option<usize>>>,

    /// Remote transfer descriptor handles
    /// Map: engine_id -> handle
    dst_xfer_handles: Arc<Mutex<HashMap<String, usize>>>,

    /// Remote agent names (returned from handshake)
    /// Map: engine_id -> agent_name
    remote_agents: Arc<Mutex<HashMap<String, String>>>,

    /// Active receiving transfers
    /// Map: request_id -> Vec<transfer_handles>
    recving_transfers: Arc<Mutex<HashMap<String, Vec<usize>>>>,

    /// Metadata for receiving transfers
    /// Map: request_id -> TransferMeta
    recving_metadata: Arc<Mutex<HashMap<String, TransferMeta>>>,

    /// Failed receive requests
    failed_recv_reqs: Arc<Mutex<HashSet<String>>>,

    /// Invalid block IDs (blocks that failed to load)
    invalid_block_ids: Arc<Mutex<HashSet<u32>>>,

    /// Successfully completed receiving transfers
    done_recving: Arc<Mutex<HashSet<String>>>,

    /// Successfully completed sending transfers
    done_sending: Arc<Mutex<HashSet<String>>>,

    /// Side channel host
    side_channel_host: String,

    /// Side channel port
    side_channel_port: u16,
}

#[pymethods]
impl ConnectorWorker {
    #[new]
    #[pyo3(signature = (engine_id, tp_rank, block_size, side_channel_host, side_channel_port, nixl_agent=None))]
    fn new(
        engine_id: String,
        tp_rank: u64,
        block_size: usize,
        side_channel_host: String,
        side_channel_port: u16,
        nixl_agent: Option<PyObject>,
    ) -> Self {
        #[cfg(feature = "nixl")]
        let agent = nixl_agent.map(|agent_obj| {
            Python::with_gil(|py| {
                agent_obj.extract::<Py<crate::NixlAgent>>(py).unwrap()
            })
        });

        Self {
            engine_id,
            tp_rank,
            block_size,
            kv_caches: Arc::new(Mutex::new(HashMap::new())),
            #[cfg(feature = "nixl")]
            nixl_agent: agent,
            src_xfer_handle: Arc::new(Mutex::new(None)),
            dst_xfer_handles: Arc::new(Mutex::new(HashMap::new())),
            remote_agents: Arc::new(Mutex::new(HashMap::new())),
            recving_transfers: Arc::new(Mutex::new(HashMap::new())),
            recving_metadata: Arc::new(Mutex::new(HashMap::new())),
            failed_recv_reqs: Arc::new(Mutex::new(HashSet::new())),
            invalid_block_ids: Arc::new(Mutex::new(HashSet::new())),
            done_recving: Arc::new(Mutex::new(HashSet::new())),
            done_sending: Arc::new(Mutex::new(HashSet::new())),
            side_channel_host,
            side_channel_port,
        }
    }

    /// Register KV caches with NIXL
    ///
    /// Takes a dict of layer_name -> tensor-like object with data_ptr() and nbytes() methods
    fn register_kv_caches(&self, kv_caches_dict: &Bound<'_, PyDict>) -> PyResult<()> {
        let mut kv_caches = self.kv_caches.lock().unwrap();

        // Extract cache information from PyTorch tensors
        for (layer_name_obj, tensor_obj) in kv_caches_dict.iter() {
            let layer_name: String = layer_name_obj.extract()?;

            // Get tensor data pointer
            let data_ptr: usize = tensor_obj.call_method0("data_ptr")?.extract()?;

            // Get tensor size in bytes
            let nbytes: usize = tensor_obj.getattr("nbytes")?.extract()?;

            // Get device (assuming CUDA tensor with .device.index)
            let device = tensor_obj.getattr("device")?;
            let dev_id: u64 = device.getattr("index")?.extract()?;

            // Get shape to calculate num_blocks
            let shape: Vec<usize> = tensor_obj.getattr("shape")?.extract()?;
            let num_blocks = if !shape.is_empty() { shape[0] } else { 0 };

            let block_len = if num_blocks > 0 {
                nbytes / num_blocks
            } else {
                0
            };

            let cache_info = KVCacheInfo {
                base_addr: data_ptr,
                size: nbytes,
                num_blocks,
                block_len,
                dev_id,
            };

            kv_caches.insert(layer_name, cache_info);
        }

        // Register with NIXL if feature is enabled
        #[cfg(feature = "nixl")]
        if let Some(agent) = &self.nixl_agent {
            self.register_with_nixl(&kv_caches)?;
        }

        Ok(())
    }

    /// Perform handshake with remote instance
    ///
    /// Returns the remote agent name
    fn nixl_handshake(
        &self,
        host: String,
        port: u16,
        remote_engine_id: String,
    ) -> PyResult<String> {
        #[cfg(feature = "nixl")]
        {
            let agent_py = self.nixl_agent.as_ref()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "NIXL agent not initialized"
                ))?;

            Python::with_gil(|py| {
                let agent = agent_py.bind(py);

                // Request metadata from remote via side channel
                let metadata_bytes = agent.call_method1(
                    "request_remote_metadata",
                    (host.clone(), port, Some(5000))
                )?.extract::<Vec<u8>>()?;

                // Add remote agent
                let agent_name = agent.call_method1(
                    "add_remote_agent",
                    (&metadata_bytes[..],)
                )?.extract::<String>()?;

                // Store agent name for this engine
                self.remote_agents.lock().unwrap().insert(
                    remote_engine_id.clone(),
                    agent_name.clone()
                );

                // Prepare remote transfer descriptors
                let kv_caches = self.kv_caches.lock().unwrap();
                let blocks_data: Vec<(usize, usize, u64)> = kv_caches
                    .values()
                    .flat_map(|info| {
                        (0..info.num_blocks).map(move |block_idx| {
                            let offset = block_idx * info.block_len;
                            (info.base_addr + offset, info.block_len, info.dev_id)
                        })
                    })
                    .collect();

                let xfer_descs_handle = agent.call_method1(
                    "get_xfer_descs",
                    (blocks_data, "VRAM")
                )?.extract::<usize>()?;

                let remote_handle = agent.call_method1(
                    "prep_xfer_dlist",
                    (&agent_name, xfer_descs_handle)
                )?.extract::<usize>()?;

                // Store remote handle
                self.dst_xfer_handles.lock().unwrap().insert(
                    remote_engine_id,
                    remote_handle
                );

                Ok(agent_name)
            })
        }

        #[cfg(not(feature = "nixl"))]
        {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "NIXL not available - build with --features nixl"
            ))
        }
    }

    /// Start loading KV from remote instances
    ///
    /// Takes metadata dict from scheduler with 'reqs_to_recv' containing request info
    fn start_load_kv(&self, metadata: &Bound<'_, PyDict>) -> PyResult<()> {
        // Extract reqs_to_recv from metadata
        let reqs_to_recv_item = metadata
            .get_item("reqs_to_recv")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Missing 'reqs_to_recv' in metadata"
            ))?;
        let reqs_to_recv = reqs_to_recv_item.downcast::<PyDict>()?;

        for (req_id_obj, req_meta_obj) in reqs_to_recv.iter() {
            let request_id: String = req_id_obj.extract()?;
            let req_meta = req_meta_obj.downcast::<PyDict>()?;

            // Extract request metadata
            let local_block_ids_item = req_meta
                .get_item("local_block_ids")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Missing local_block_ids"
                ))?;
            let local_block_ids: Vec<u32> = local_block_ids_item.extract()?;

            let remote_block_ids_item = req_meta
                .get_item("remote_block_ids")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Missing remote_block_ids"
                ))?;
            let remote_block_ids: Vec<u32> = remote_block_ids_item.extract()?;

            // Skip if no blocks to transfer
            if local_block_ids.is_empty() || remote_block_ids.is_empty() {
                continue;
            }

            let remote_engine_id_item = req_meta
                .get_item("remote_engine_id")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Missing remote_engine_id"
                ))?;
            let remote_engine_id: String = remote_engine_id_item.extract()?;

            let remote_host_item = req_meta
                .get_item("remote_host")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Missing remote_host"
                ))?;
            let remote_host: String = remote_host_item.extract()?;

            let remote_port_item = req_meta
                .get_item("remote_port")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Missing remote_port"
                ))?;
            let remote_port: u16 = remote_port_item.extract()?;

            // Store metadata
            let transfer_meta = TransferMeta {
                request_id: request_id.clone(),
                local_block_ids: local_block_ids.clone(),
                remote_block_ids: remote_block_ids.clone(),
                remote_engine_id: remote_engine_id.clone(),
                remote_host: remote_host.clone(),
                remote_port,
            };

            self.recving_metadata.lock().unwrap().insert(
                request_id.clone(),
                transfer_meta
            );

            // Perform handshake if needed
            let remote_agents = self.remote_agents.lock().unwrap();
            if !remote_agents.contains_key(&remote_engine_id) {
                drop(remote_agents);
                self.nixl_handshake(
                    remote_host,
                    remote_port,
                    remote_engine_id.clone()
                )?;
            }

            // Initiate transfer
            #[cfg(feature = "nixl")]
            self.read_blocks(&request_id, &local_block_ids, &remote_block_ids, &remote_engine_id)?;
        }

        Ok(())
    }

    /// Poll for finished transfers
    ///
    /// Returns (done_sending, done_recving) sets of request IDs
    fn get_finished<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PySet>, Bound<'py, PySet>)> {
        #[cfg(feature = "nixl")]
        {
            let agent_py = self.nixl_agent.as_ref()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "NIXL agent not initialized"
                ))?;

            let agent = agent_py.bind(py);

            // Check for sending confirmations (notifications from remote)
            let notifs_result = agent.call_method0("get_new_notifs")?;
            let notifs_dict = notifs_result.downcast::<PyDict>()?;

            let mut done_sending = self.done_sending.lock().unwrap();
            for (_agent_name, notifs_list_obj) in notifs_dict.iter() {
                let notifs_list = notifs_list_obj.downcast::<PyList>()?;
                for notif_obj in notifs_list.iter() {
                    let notif_bytes: Vec<u8> = notif_obj.extract()?;
                    let request_id = String::from_utf8(notif_bytes)
                        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Invalid notification message"
                        ))?;
                    done_sending.insert(request_id);
                }
            }

            // Poll active receiving transfers
            let mut recving_transfers = self.recving_transfers.lock().unwrap();
            let mut completed_requests = Vec::new();

            for (request_id, handles) in recving_transfers.iter() {
                let mut all_done = true;

                for &handle in handles {
                    let state = agent.call_method1("check_xfer_state", (handle,))?
                        .extract::<String>()?;

                    if state == "DONE" {
                        // Get telemetry and release handle
                        let _telemetry = agent.call_method1("get_xfer_telemetry", (handle,))?;
                        agent.call_method1("release_xfer_handle", (handle,))?;
                    } else if state != "PROC" {
                        // Transfer failed
                        all_done = false;
                        self.failed_recv_reqs.lock().unwrap().insert(request_id.clone());

                        // Mark blocks as invalid
                        if let Some(meta) = self.recving_metadata.lock().unwrap().get(request_id) {
                            let mut invalid_blocks = self.invalid_block_ids.lock().unwrap();
                            for &block_id in &meta.local_block_ids {
                                invalid_blocks.insert(block_id);
                            }
                        }
                        break;
                    } else {
                        // Still in progress
                        all_done = false;
                    }
                }

                if all_done {
                    completed_requests.push(request_id.clone());
                }
            }

            // Remove completed requests
            let mut done_recving = self.done_recving.lock().unwrap();
            for request_id in completed_requests {
                recving_transfers.remove(&request_id);
                self.recving_metadata.lock().unwrap().remove(&request_id);
                done_recving.insert(request_id);
            }

            // Create Python sets
            let done_sending_set = PySet::new_bound(py, &*done_sending)?;
            let done_recving_set = PySet::new_bound(py, &*done_recving)?;

            // Clear the sets for next call
            done_sending.clear();
            done_recving.clear();

            Ok((done_sending_set, done_recving_set))
        }

        #[cfg(not(feature = "nixl"))]
        {
            // Return empty sets when NIXL is not available
            let empty_sending = PySet::empty_bound(py)?;
            let empty_recving = PySet::empty_bound(py)?;
            Ok((empty_sending, empty_recving))
        }
    }

    /// Get block IDs that failed to load
    fn get_block_ids_with_load_errors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        let invalid_blocks = self.invalid_block_ids.lock().unwrap();
        let block_set = PySet::new_bound(py, &*invalid_blocks)?;
        Ok(block_set)
    }

    /// Clear error tracking (for testing)
    fn clear_errors(&self) -> PyResult<()> {
        self.invalid_block_ids.lock().unwrap().clear();
        self.failed_recv_reqs.lock().unwrap().clear();
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "ConnectorWorker(engine={}, tp_rank={}, active_transfers={})",
            self.engine_id,
            self.tp_rank,
            self.recving_transfers.lock().unwrap().len()
        )
    }
}

// Internal helper methods (not exposed to Python)
impl ConnectorWorker {
    /// Internal method to register memory with NIXL
    #[cfg(feature = "nixl")]
    fn register_with_nixl(&self, kv_caches: &HashMap<String, KVCacheInfo>) -> PyResult<()> {
        let agent_py = self.nixl_agent.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "NIXL agent not initialized"
            ))?;

        // Collect all cache data for registration
        let cache_data: Vec<(usize, usize, u64)> = kv_caches
            .values()
            .map(|info| (info.base_addr, info.size, info.dev_id))
            .collect();

        Python::with_gil(|py| {
            let agent = agent_py.bind(py);

            // Get registration descriptors
            let reg_descs_handle = agent.call_method1(
                "get_reg_descs",
                (cache_data, "VRAM")
            )?.extract::<usize>()?;

            // Register memory with UCX backend
            agent.call_method1(
                "register_memory",
                (reg_descs_handle, vec!["UCX"])
            )?;

            // Get transfer descriptors for local operations
            let blocks_data: Vec<(usize, usize, u64)> = kv_caches
                .values()
                .flat_map(|info| {
                    (0..info.num_blocks).map(move |block_idx| {
                        let offset = block_idx * info.block_len;
                        (info.base_addr + offset, info.block_len, info.dev_id)
                    })
                })
                .collect();

            let xfer_descs_handle = agent.call_method1(
                "get_xfer_descs",
                (blocks_data, "VRAM")
            )?.extract::<usize>()?;

            // Store for later use
            *self.src_xfer_handle.lock().unwrap() = Some(xfer_descs_handle);

            Ok::<(), PyErr>(())
        })
    }

    /// Internal method to read blocks from remote
    #[cfg(feature = "nixl")]
    fn read_blocks(
        &self,
        request_id: &str,
        local_block_ids: &[u32],
        remote_block_ids: &[u32],
        remote_engine_id: &str,
    ) -> PyResult<()> {
        let agent_py = self.nixl_agent.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "NIXL agent not initialized"
            ))?;

        Python::with_gil(|py| {
            let agent = agent_py.bind(py);

            // Get local and remote handles
            let local_handle = *self.src_xfer_handle.lock().unwrap()
                .as_ref()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Local transfer handle not initialized"
                ))?;

            let remote_handle = *self.dst_xfer_handles.lock().unwrap()
                .get(remote_engine_id)
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("No remote handle for engine {}", remote_engine_id)
                ))?;

            // Convert block IDs to descriptor IDs (i32 for NIXL API)
            let local_ids: Vec<i32> = local_block_ids.iter().map(|&id| id as i32).collect();
            let remote_ids: Vec<i32> = remote_block_ids.iter().map(|&id| id as i32).collect();

            // Create notification message with request_id
            let notif_msg = request_id.as_bytes();

            // Make prepared transfer
            let xfer_handle = agent.call_method(
                "make_prepped_xfer",
                (
                    "READ",
                    local_handle,
                    local_ids,
                    remote_handle,
                    remote_ids,
                ),
                Some(&[("notif_msg", notif_msg)].into_py_dict_bound(py))
            )?.extract::<usize>()?;

            // Start the transfer
            agent.call_method1("transfer", (xfer_handle,))?;

            // Store transfer handle
            self.recving_transfers.lock().unwrap()
                .entry(request_id.to_string())
                .or_insert_with(Vec::new)
                .push(xfer_handle);

            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_creation() {
        let worker = ConnectorWorker::new(
            "test_engine".to_string(),
            0,
            16,
            "0.0.0.0".to_string(),
            8000,
            None,
        );

        assert_eq!(worker.engine_id, "test_engine");
        assert_eq!(worker.tp_rank, 0);
        assert_eq!(worker.block_size, 16);
        assert!(worker.kv_caches.lock().unwrap().is_empty());
    }

    #[test]
    fn test_kv_cache_info() {
        let info = KVCacheInfo {
            base_addr: 0x1000,
            size: 4096,
            num_blocks: 16,
            block_len: 256,
            dev_id: 0,
        };

        assert_eq!(info.base_addr, 0x1000);
        assert_eq!(info.num_blocks, 16);
        assert_eq!(info.block_len, 256);
    }
}
