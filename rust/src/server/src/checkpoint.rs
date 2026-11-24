// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The process checkpoint lifecycle, shared by the HTTP routes and gRPC Control.

use vllm_engine_core_client::EngineCoreClient;
use vllm_engine_core_client::protocol::utility::{CheckpointState, PauseMode};

/// An engine utility call that failed, and which one.
pub(crate) struct CallError {
    pub method: &'static str,
    pub error: vllm_engine_core_client::Error,
}

/// What a prepare or restore request did.
pub(crate) enum Transition {
    /// Started, already under way, or already done; the state after the call.
    Accepted(CheckpointState, Option<String>),
    /// An engine refused to prepare and nothing changed.
    Refused(String),
    /// Not possible from this state.
    Invalid(CheckpointState, Option<String>),
}

pub(crate) async fn state(
    client: &EngineCoreClient,
) -> Result<(CheckpointState, Option<String>), CallError> {
    client.checkpoint_state().await.map_err(|error| CallError {
        method: "checkpoint_state",
        error,
    })
}

/// Drain with `mode` and release what a process checkpoint must not contain.
/// Repeating while preparing or prepared is a no-op.
pub(crate) async fn prepare(
    client: &EngineCoreClient,
    mode: PauseMode,
) -> Result<Transition, CallError> {
    let (state_now, error) = state(client).await?;
    match state_now {
        CheckpointState::Running => {
            let refusal = client.checkpoint_prepare(mode).await.map_err(|error| CallError {
                method: "checkpoint_prepare",
                error,
            })?;
            if let Some(refusal) = refusal {
                return Ok(Transition::Refused(refusal));
            }
            let (state_now, error) = state(client).await?;
            Ok(Transition::Accepted(state_now, error))
        }
        CheckpointState::Restoring | CheckpointState::Failed => {
            Ok(Transition::Invalid(state_now, error))
        }
        CheckpointState::Preparing | CheckpointState::Prepared => {
            Ok(Transition::Accepted(state_now, error))
        }
    }
}

/// Bring back what preparing released and serve, in the restored process or in
/// place. Orchestrators may deliver this more than once, so a restore in
/// progress or done is accepted.
pub(crate) async fn restore(client: &EngineCoreClient) -> Result<Transition, CallError> {
    let (state_now, error) = state(client).await?;
    match state_now {
        CheckpointState::Prepared => {
            client.checkpoint_restore().await.map_err(|error| CallError {
                method: "checkpoint_restore",
                error,
            })?;
            let (state_now, error) = state(client).await?;
            Ok(Transition::Accepted(state_now, error))
        }
        CheckpointState::Preparing | CheckpointState::Failed => {
            Ok(Transition::Invalid(state_now, error))
        }
        CheckpointState::Running | CheckpointState::Restoring => {
            Ok(Transition::Accepted(state_now, error))
        }
    }
}
