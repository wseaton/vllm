// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Python bindings for NIXL Rust library
//!
//! Provides a Python-compatible API wrapper around nixl-sys that matches
//! the existing nixl Python bindings interface.
//!
//! Uses NIXL's safe Rust API with the NixlDescriptor trait pattern for
//! type-safe memory registration and transfer operations.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

pub mod side_channel;
pub mod scheduler;
pub mod worker;

use side_channel::SideChannel;
use scheduler::{ConnectorScheduler, RequestMeta};
use worker::ConnectorWorker;

#[cfg(feature = "nixl")]
use nixl_sys::{
    Agent, AgentConfig, MemType, NotificationMap,
    OptArgs, RegDescList, XferDescList, XferDlistHandle, XferOp, XferRequest,
    XferStatus, ThreadSync, MemoryRegion, NixlDescriptor, RegistrationHandle,
};

#[cfg(feature = "nixl")]
mod descriptors {
    use super::*;

    /// Memory descriptor that implements NixlDescriptor trait
    /// used for NIXL memory registration
    #[derive(Debug)]
    pub struct VllmMemoryDescriptor {
        ptr: usize,
        size: usize,
        dev_id: u64,
        mem_type: MemType,
    }

    impl VllmMemoryDescriptor {
        pub fn new(ptr: usize, size: usize, dev_id: u64, mem_type: MemType) -> Self {
            Self {
                ptr,
                size,
                dev_id,
                mem_type,
            }
        }
    }

    impl MemoryRegion for VllmMemoryDescriptor {
        unsafe fn as_ptr(&self) -> *const u8 {
            self.ptr as *const u8
        }

        fn size(&self) -> usize {
            self.size
        }
    }

    impl NixlDescriptor for VllmMemoryDescriptor {
        fn mem_type(&self) -> MemType {
            self.mem_type
        }

        fn device_id(&self) -> u64 {
            self.dev_id
        }
    }
}

/// Descriptor data stored for later registration
#[cfg(feature = "nixl")]
#[derive(Debug, Clone)]
struct DescriptorData {
    addr: usize,
    size: usize,
    dev_id: u64,
    mem_type: MemType,
}

/// thread-safe wrapper for XferDlistHandle
/// SAFETY: NIXL agent is thread-safe, and these handles are valid across threads
#[cfg(feature = "nixl")]
struct SendableXferDlistHandle(XferDlistHandle);

#[cfg(feature = "nixl")]
unsafe impl Send for SendableXferDlistHandle {}
#[cfg(feature = "nixl")]
unsafe impl Sync for SendableXferDlistHandle {}

/// thread-safe wrapper for XferRequest
/// SAFETY: NIXL agent is thread-safe, and these requests are valid across threads
#[cfg(feature = "nixl")]
struct SendableXferRequest(XferRequest);

#[cfg(feature = "nixl")]
unsafe impl Send for SendableXferRequest {}
#[cfg(feature = "nixl")]
unsafe impl Sync for SendableXferRequest {}

/// Python wrapper for NIXL Agent
///
/// matches the API of nixl._api.nixl_agent from Python bindings
#[pyclass]
struct NixlAgent {
    #[cfg(feature = "nixl")]
    agent: Arc<Mutex<Agent>>,
    #[cfg(feature = "nixl")]
    xfer_desc_data: Arc<Mutex<HashMap<usize, (Vec<(usize, usize, u64)>, MemType)>>>,
    #[cfg(feature = "nixl")]
    xfer_dlist_handles: Arc<Mutex<HashMap<usize, SendableXferDlistHandle>>>,
    #[cfg(feature = "nixl")]
    xfer_requests: Arc<Mutex<HashMap<usize, SendableXferRequest>>>,
    #[cfg(feature = "nixl")]
    descriptor_data: Arc<Mutex<HashMap<usize, Vec<DescriptorData>>>>,
    #[cfg(feature = "nixl")]
    reg_handles: Arc<Mutex<HashMap<usize, Vec<RegistrationHandle>>>>,
    side_channel: Arc<Mutex<Box<dyn side_channel::SideChannel>>>,
    #[cfg(not(feature = "nixl"))]
    _phantom: std::marker::PhantomData<()>,
}

#[pymethods]
impl NixlAgent {
    #[new]
    #[pyo3(signature = (name, config=None))]
    fn new(name: &str, config: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        #[cfg(feature = "nixl")]
        {
            let agent = if let Some(cfg_dict) = config {
                let mut agent_config = AgentConfig::default();
                agent_config.capture_telemetry = true;

                if let Ok(Some(enable_prog_thread)) = cfg_dict.get_item("enable_prog_thread") {
                    agent_config.enable_prog_thread = enable_prog_thread.extract()?;
                }
                if let Ok(Some(num_workers)) = cfg_dict.get_item("num_workers") {
                    agent_config.num_workers = num_workers.extract()?;
                }

                Agent::new_configured(name, &agent_config)
            } else {
                Agent::new(name)
            }.map_err(|e| PyRuntimeError::new_err(format!("Failed to create agent: {:?}", e)))?;

            Ok(Self {
                agent: Arc::new(Mutex::new(agent)),
                xfer_desc_data: Arc::new(Mutex::new(HashMap::new())),
                xfer_dlist_handles: Arc::new(Mutex::new(HashMap::new())),
                xfer_requests: Arc::new(Mutex::new(HashMap::new())),
                descriptor_data: Arc::new(Mutex::new(HashMap::new())),
                reg_handles: Arc::new(Mutex::new(HashMap::new())),
                side_channel: Arc::new(Mutex::new(Box::new(
                    side_channel::tcp::TcpSideChannel::new()
                ))),
            })
        }

        #[cfg(not(feature = "nixl"))]
        {
            Err(PyRuntimeError::new_err(
                "NIXL Rust bindings not available. Build with --features nixl"
            ))
        }
    }

    /// Get agent metadata as bytes for handshake
    fn get_agent_metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        #[cfg(feature = "nixl")]
        {
            let agent = self.agent.lock().unwrap();
            let metadata = agent.get_local_md()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to get metadata: {:?}", e)))?;
            Ok(PyBytes::new_bound(py, &metadata))
        }

        #[cfg(not(feature = "nixl"))]
        {
            Err(PyRuntimeError::new_err("NIXL not available"))
        }
    }

    /// Add a remote agent by loading its metadata
    fn add_remote_agent(&self, metadata: &[u8]) -> PyResult<String> {
        #[cfg(feature = "nixl")]
        {
            let agent = self.agent.lock().unwrap();
            agent.load_remote_md(metadata)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to load remote metadata: {:?}", e)))
        }

        #[cfg(not(feature = "nixl"))]
        {
            Err(PyRuntimeError::new_err("NIXL not available"))
        }
    }

    /// Remove a remote agent
    fn remove_remote_agent(&self, agent_name: &str) -> PyResult<()> {
        #[cfg(feature = "nixl")]
        {
            let agent = self.agent.lock().unwrap();
            agent.invalidate_remote_md(agent_name)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to invalidate remote: {:?}", e)))
        }

        #[cfg(not(feature = "nixl"))]
        {
            Err(PyRuntimeError::new_err("NIXL not available"))
        }
    }

    /// Get registration descriptors
    ///
    /// Stores descriptor data for later registration using NixlDescriptor trait
    fn get_reg_descs(&self, cache_data: Vec<(usize, usize, u64)>, memory_type: &str) -> PyResult<usize> {
        #[cfg(feature = "nixl")]
        {
            let mem_type = match memory_type {
                "DRAM" => MemType::Dram,
                "VRAM" => MemType::Vram,
                _ => return Err(PyValueError::new_err(format!("Unknown memory type: {}", memory_type))),
            };

            // store descriptor data for later use in register_memory
            let descriptors: Vec<DescriptorData> = cache_data
                .into_iter()
                .map(|(addr, size, dev_id)| DescriptorData {
                    addr,
                    size,
                    dev_id,
                    mem_type,
                })
                .collect();

            // generate a unique handle ID
            let handle_id = Arc::as_ptr(&self.agent) as usize + descriptors.len();
            self.descriptor_data.lock().unwrap().insert(handle_id, descriptors);

            Ok(handle_id)
        }

        #[cfg(not(feature = "nixl"))]
        {
            Err(PyRuntimeError::new_err("NIXL not available"))
        }
    }

    /// Register memory with NIXL
    ///
    /// Uses NixlDescriptor trait for safe, idiomatic registration
    fn register_memory(&self, reg_descs_handle: usize, backends: Vec<String>) -> PyResult<()> {
        #[cfg(feature = "nixl")]
        {
            use descriptors::VllmMemoryDescriptor;

            let agent = self.agent.lock().unwrap();

            // retrieve stored descriptor data
            let descriptors = self.descriptor_data.lock().unwrap()
                .get(&reg_descs_handle)
                .ok_or_else(|| PyValueError::new_err("Invalid reg_descs handle"))?
                .clone();

            // create opt_args with backends if provided
            let opt_args = if !backends.is_empty() {
                let mut args = OptArgs::new()
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to create opt args: {:?}", e)))?;

                for backend_name in backends {
                    let (_, params) = agent.get_plugin_params(&backend_name)
                        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get plugin params: {:?}", e)))?;

                    let backend = agent.create_backend(&backend_name, &params)
                        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create backend: {:?}", e)))?;

                    args.add_backend(&backend)
                        .map_err(|e| PyRuntimeError::new_err(format!("Failed to add backend: {:?}", e)))?;
                }
                Some(args)
            } else {
                None
            };

            let mut reg_handles = Vec::new();

            // register each descriptor individually using the NixlDescriptor trait
            for desc_data in descriptors {
                let descriptor = VllmMemoryDescriptor::new(
                    desc_data.addr,
                    desc_data.size,
                    desc_data.dev_id,
                    desc_data.mem_type,
                );

                let handle = agent.register_memory(&descriptor, opt_args.as_ref())
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to register memory: {:?}", e)))?;

                reg_handles.push(handle);
            }

            // store handles for cleanup
            self.reg_handles.lock().unwrap().insert(reg_descs_handle, reg_handles);

            Ok(())
        }

        #[cfg(not(feature = "nixl"))]
        {
            Err(PyRuntimeError::new_err("NIXL not available"))
        }
    }

    /// Deregister memory
    ///
    /// Uses stored RegistrationHandles which automatically deregister on drop
    fn deregister_memory(&self, reg_descs_handle: usize) -> PyResult<()> {
        #[cfg(feature = "nixl")]
        {
            // remove handles from map - they'll deregister automatically when dropped
            if let Some(mut handles) = self.reg_handles.lock().unwrap().remove(&reg_descs_handle) {
                // explicitly call deregister on each handle
                for handle in handles.iter_mut() {
                    handle.deregister()
                        .map_err(|e| PyRuntimeError::new_err(format!("Failed to deregister memory: {:?}", e)))?;
                }
            }
            Ok(())
        }

        #[cfg(not(feature = "nixl"))]
        {
            Err(PyRuntimeError::new_err("NIXL not available"))
        }
    }

    /// Get transfer descriptors
    fn get_xfer_descs(&self, blocks_data: Vec<(usize, usize, u64)>, memory_type: &str) -> PyResult<usize> {
        #[cfg(feature = "nixl")]
        {
            let mem_type = match memory_type {
                "DRAM" => MemType::Dram,
                "VRAM" => MemType::Vram,
                _ => return Err(PyValueError::new_err(format!("Unknown memory type: {}", memory_type))),
            };

            // store descriptor data for later use in prep_xfer_dlist
            let handle_id = self.xfer_desc_data.lock().unwrap().len() + 1;
            self.xfer_desc_data.lock().unwrap().insert(handle_id, (blocks_data, mem_type));

            Ok(handle_id)
        }

        #[cfg(not(feature = "nixl"))]
        {
            Err(PyRuntimeError::new_err("NIXL not available"))
        }
    }

    /// Prepare transfer descriptor list
    fn prep_xfer_dlist(&self, agent_name: &str, xfer_descs_handle: usize) -> PyResult<usize> {
        #[cfg(feature = "nixl")]
        {
            let agent = self.agent.lock().unwrap();
            let xfer_desc_data_map = self.xfer_desc_data.lock().unwrap();

            let (blocks_data, mem_type) = xfer_desc_data_map
                .get(&xfer_descs_handle)
                .ok_or_else(|| PyValueError::new_err("Invalid xfer_descs handle"))?;

            // create XferDescList from stored data
            let mut xfer_dlist = XferDescList::new(*mem_type)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create xfer desc list: {:?}", e)))?;

            for &(addr, size, dev_id) in blocks_data {
                xfer_dlist.add_desc(addr, size, dev_id)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to add descriptor: {:?}", e)))?;
            }

            let dlist_handle = agent.prepare_xfer_dlist(agent_name, &xfer_dlist, None)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to prep xfer dlist: {:?}", e)))?;

            // release lock before acquiring another
            drop(xfer_desc_data_map);

            // generate unique handle and store
            let handle_id = self.xfer_dlist_handles.lock().unwrap().len() + 1;
            self.xfer_dlist_handles.lock().unwrap().insert(handle_id, SendableXferDlistHandle(dlist_handle));

            Ok(handle_id)
        }

        #[cfg(not(feature = "nixl"))]
        {
            Err(PyRuntimeError::new_err("NIXL not available"))
        }
    }

    /// Release a prepared descriptor list handle
    fn release_dlist_handle(&self, handle_id: usize) -> PyResult<()> {
        #[cfg(feature = "nixl")]
        {
            self.xfer_dlist_handles.lock().unwrap().remove(&handle_id);
            Ok(())
        }

        #[cfg(not(feature = "nixl"))]
        {
            Err(PyRuntimeError::new_err("NIXL not available"))
        }
    }

    /// Make a prepared transfer
    #[pyo3(signature = (op, local_handle, local_ids, remote_handle, remote_ids, notif_msg=None))]
    fn make_prepped_xfer(
        &self,
        op: &str,
        local_handle: usize,
        local_ids: Vec<i32>,
        remote_handle: usize,
        remote_ids: Vec<i32>,
        notif_msg: Option<&[u8]>,
    ) -> PyResult<usize> {
        #[cfg(feature = "nixl")]
        {
            let agent = self.agent.lock().unwrap();
            let xfer_op = match op {
                "READ" => XferOp::Read,
                "WRITE" => XferOp::Write,
                _ => return Err(PyValueError::new_err(format!("Unknown operation: {}", op))),
            };

            let dlist_handles = self.xfer_dlist_handles.lock().unwrap();

            let local_dlist = &dlist_handles
                .get(&local_handle)
                .ok_or_else(|| PyValueError::new_err("Invalid local handle"))?
                .0;

            let remote_dlist = &dlist_handles
                .get(&remote_handle)
                .ok_or_else(|| PyValueError::new_err("Invalid remote handle"))?
                .0;

            let opt_args = if let Some(msg) = notif_msg {
                let mut args = OptArgs::new()
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to create opt args: {:?}", e)))?;
                args.set_notification_message(msg)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to set notification: {:?}", e)))?;
                args.set_has_notification(true)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to enable notification: {:?}", e)))?;
                Some(args)
            } else {
                None
            };

            let xfer_req = agent.make_xfer_req(
                xfer_op,
                local_dlist,
                &local_ids,
                remote_dlist,
                &remote_ids,
                opt_args.as_ref(),
            ).map_err(|e| PyRuntimeError::new_err(format!("Failed to create transfer: {:?}", e)))?;

            // release lock before acquiring another
            drop(dlist_handles);

            // store the request
            let req_id = self.xfer_requests.lock().unwrap().len() + 1;
            self.xfer_requests.lock().unwrap().insert(req_id, SendableXferRequest(xfer_req));

            Ok(req_id)
        }

        #[cfg(not(feature = "nixl"))]
        {
            Err(PyRuntimeError::new_err("NIXL not available"))
        }
    }

    /// Initiate a transfer
    fn transfer(&self, xfer_handle: usize) -> PyResult<()> {
        #[cfg(feature = "nixl")]
        {
            let agent = self.agent.lock().unwrap();
            let xfer_requests = self.xfer_requests.lock().unwrap();

            let xfer_req = &xfer_requests
                .get(&xfer_handle)
                .ok_or_else(|| PyValueError::new_err("Invalid transfer handle"))?
                .0;

            agent.post_xfer_req(xfer_req, None)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to post transfer: {:?}", e)))?;

            Ok(())
        }

        #[cfg(not(feature = "nixl"))]
        {
            Err(PyRuntimeError::new_err("NIXL not available"))
        }
    }

    /// Check transfer state
    fn check_xfer_state(&self, xfer_handle: usize) -> PyResult<String> {
        #[cfg(feature = "nixl")]
        {
            let agent = self.agent.lock().unwrap();
            let xfer_requests = self.xfer_requests.lock().unwrap();

            let xfer_req = &xfer_requests
                .get(&xfer_handle)
                .ok_or_else(|| PyValueError::new_err("Invalid transfer handle"))?
                .0;

            let status = agent.get_xfer_status(xfer_req)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to check status: {:?}", e)))?;

            Ok(match status {
                XferStatus::Success => "DONE".to_string(),
                XferStatus::InProgress => "PROC".to_string(),
            })
        }

        #[cfg(not(feature = "nixl"))]
        {
            Err(PyRuntimeError::new_err("NIXL not available"))
        }
    }

    /// Get transfer telemetry
    fn get_xfer_telemetry<'py>(&self, py: Python<'py>, xfer_handle: usize) -> PyResult<Bound<'py, PyDict>> {
        #[cfg(feature = "nixl")]
        {
            let xfer_requests = self.xfer_requests.lock().unwrap();

            let xfer_req = &xfer_requests
                .get(&xfer_handle)
                .ok_or_else(|| PyValueError::new_err("Invalid transfer handle"))?
                .0;

            let telemetry = xfer_req.get_telemetry()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to get telemetry: {:?}", e)))?;

            let dict = PyDict::new_bound(py);
            dict.set_item("xferDuration", telemetry.xfer_duration_us)?;
            dict.set_item("postDuration", telemetry.post_duration_us)?;
            dict.set_item("totalBytes", telemetry.total_bytes)?;
            dict.set_item("descCount", telemetry.desc_count)?;

            Ok(dict)
        }

        #[cfg(not(feature = "nixl"))]
        {
            Err(PyRuntimeError::new_err("NIXL not available"))
        }
    }

    /// Release a transfer handle
    fn release_xfer_handle(&self, xfer_handle: usize) -> PyResult<()> {
        #[cfg(feature = "nixl")]
        {
            self.xfer_requests.lock().unwrap().remove(&xfer_handle);
            Ok(())
        }

        #[cfg(not(feature = "nixl"))]
        {
            Err(PyRuntimeError::new_err("NIXL not available"))
        }
    }

    /// Send a notification to a remote agent
    fn send_notif(&self, agent_name: &str, notif_msg: &[u8]) -> PyResult<()> {
        #[cfg(feature = "nixl")]
        {
            let agent = self.agent.lock().unwrap();
            agent.send_notification(agent_name, notif_msg, None)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to send notification: {:?}", e)))
        }

        #[cfg(not(feature = "nixl"))]
        {
            Err(PyRuntimeError::new_err("NIXL not available"))
        }
    }

    /// Get new notifications
    fn get_new_notifs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        #[cfg(feature = "nixl")]
        {
            let agent = self.agent.lock().unwrap();
            let mut notif_map = NotificationMap::new()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create notif map: {:?}", e)))?;

            agent.get_notifications(&mut notif_map, None)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to get notifications: {:?}", e)))?;

            let notifications = notif_map.take_notifs()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to take notifications: {:?}", e)))?;

            let result = PyDict::new_bound(py);
            for (agent_name, notifs) in notifications {
                let notifs_list = PyList::empty_bound(py);
                for notif in notifs {
                    notifs_list.append(PyBytes::new_bound(py, notif.as_bytes()))?;
                }
                result.set_item(agent_name, notifs_list)?;
            }

            Ok(result)
        }

        #[cfg(not(feature = "nixl"))]
        {
            Err(PyRuntimeError::new_err("NIXL not available"))
        }
    }

    /// Start side channel listener for metadata exchange
    ///
    /// # Arguments
    /// * `port` - port to listen on
    /// * `metadata` - agent metadata bytes to serve
    fn start_side_channel_listener(&self, port: u16, metadata: &[u8]) -> PyResult<()> {
        let mut channel = self.side_channel.lock().unwrap();
        channel
            .start_listener(port, metadata.to_vec())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to start listener: {}", e)))
    }

    /// Request metadata from remote agent via side channel
    ///
    /// # Arguments
    /// * `host` - hostname or IP
    /// * `port` - port number
    /// * `timeout_ms` - timeout in milliseconds (default 5000)
    #[pyo3(signature = (host, port, timeout_ms=None))]
    fn request_remote_metadata<'py>(
        &self,
        py: Python<'py>,
        host: &str,
        port: u16,
        timeout_ms: Option<u64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let timeout = timeout_ms.unwrap_or(5000);
        let channel = self.side_channel.lock().unwrap();
        let metadata = channel
            .request_metadata(host, port, timeout)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to request metadata: {}", e)))?;
        Ok(PyBytes::new_bound(py, &metadata))
    }

    /// Stop side channel listener
    fn stop_side_channel_listener(&self) -> PyResult<()> {
        let mut channel = self.side_channel.lock().unwrap();
        channel
            .shutdown()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to shutdown listener: {}", e)))
    }

    /// Check if side channel listener is running
    fn is_side_channel_running(&self) -> PyResult<bool> {
        let channel = self.side_channel.lock().unwrap();
        Ok(channel.is_running())
    }
}

/// nixl_agent_config helper function (matches Python API)
#[pyfunction]
#[pyo3(signature = (backends=None, num_threads=None))]
fn nixl_agent_config(
    py: Python<'_>,
    backends: Option<Vec<String>>,
    num_threads: Option<u32>,
) -> PyResult<Bound<'_, PyDict>> {
    let config = PyDict::new_bound(py);
    config.set_item("enable_prog_thread", true)?;
    config.set_item("capture_telemetry", true)?;

    if let Some(n) = num_threads {
        config.set_item("num_workers", n)?;
    }

    if let Some(_backends) = backends {
        // backends configuration - store for later use
        // the Python code will pass this to Agent constructor
    }

    Ok(config)
}

/// Python wrapper for TcpSideChannel (for testing without NIXL)
#[pyclass]
struct TcpSideChannel {
    inner: Arc<Mutex<side_channel::tcp::TcpSideChannel>>,
}

#[pymethods]
impl TcpSideChannel {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(side_channel::tcp::TcpSideChannel::new())),
        }
    }

    /// start listener on specified port
    fn start_listener(&self, port: u16, metadata: &[u8]) -> PyResult<()> {
        let mut channel = self.inner.lock().unwrap();
        channel
            .start_listener(port, metadata.to_vec())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to start listener: {}", e)))
    }

    /// request metadata from remote host
    #[pyo3(signature = (host, port, timeout_ms=None))]
    fn request_metadata<'py>(
        &self,
        py: Python<'py>,
        host: &str,
        port: u16,
        timeout_ms: Option<u64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let timeout = timeout_ms.unwrap_or(5000);
        let channel = self.inner.lock().unwrap();
        let metadata = channel
            .request_metadata(host, port, timeout)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to request metadata: {}", e)))?;
        Ok(PyBytes::new_bound(py, &metadata))
    }

    /// shutdown the listener
    fn shutdown(&self) -> PyResult<()> {
        let mut channel = self.inner.lock().unwrap();
        channel
            .shutdown()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to shutdown: {}", e)))
    }

    /// check if listener is running
    fn is_running(&self) -> PyResult<bool> {
        let channel = self.inner.lock().unwrap();
        Ok(channel.is_running())
    }
}

/// vLLM NIXL Rust bindings module
#[pymodule]
fn vllm_nixl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NixlAgent>()?;
    m.add_class::<TcpSideChannel>()?;
    m.add_class::<ConnectorScheduler>()?;
    m.add_class::<ConnectorWorker>()?;
    m.add_class::<RequestMeta>()?;
    m.add_function(wrap_pyfunction!(nixl_agent_config, m)?)?;
    Ok(())
}
