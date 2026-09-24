// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::Json;
use axum::extract::{Request, State};
use axum::http::header::AUTHORIZATION;
use axum::http::{HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use openssl::error::ErrorStack;
use serde_json::json;
use tracing::error;

use crate::state::{ApiKeyHash, AppState, hash_api_key};

const GUARDED_PREFIXES: &[&str] = &["/v1", "/v2", "/inference"];

/// Authenticate guarded HTTP routes with an OpenAI-compatible bearer token.
///
/// Mirrors Python `AuthenticationMiddleware`: OPTIONS requests and non-guarded
/// helper endpoints such as `/health` are allowed through without a token.
pub async fn authenticate_api_key(
    State(state): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Response {
    if req.method() == Method::OPTIONS || !requires_auth(req.uri().path()) {
        return next.run(req).await;
    }

    match verify_token(req.headers().get(AUTHORIZATION), state.api_key_hashes()) {
        Ok(true) => next.run(req).await,
        Ok(false) => (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "error": "Unauthorized" })),
        )
            .into_response(),
        Err(err) => {
            error!(error = %err, "failed to hash bearer token");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": "Internal Server Error" })),
            )
                .into_response()
        }
    }
}

fn requires_auth(path: &str) -> bool {
    GUARDED_PREFIXES.iter().any(|prefix| path.starts_with(prefix))
}

fn verify_token(
    authorization: Option<&HeaderValue>,
    api_key_hashes: &[ApiKeyHash],
) -> Result<bool, ErrorStack> {
    let Some(authorization) = authorization else {
        return Ok(false);
    };
    let Ok(authorization) = authorization.to_str() else {
        return Ok(false);
    };
    let Some((scheme, token)) = authorization.split_once(' ') else {
        return Ok(false);
    };
    if !scheme.eq_ignore_ascii_case("bearer") {
        return Ok(false);
    }

    let token_hash = hash_api_key(token)?;
    let mut token_match = false;
    for api_key_hash in api_key_hashes {
        token_match |= constant_time_eq(&token_hash, api_key_hash);
    }
    Ok(token_match)
}

fn constant_time_eq(left: &ApiKeyHash, right: &ApiKeyHash) -> bool {
    use subtle::ConstantTimeEq;

    bool::from(left.ct_eq(right))
}

#[cfg(test)]
mod tests {
    use crate::middleware::auth::constant_time_eq;
    use crate::state::hash_api_key;

    #[test]
    fn hash_api_key_matches_sha256_known_answer() {
        assert_eq!(
            hash_api_key("abc").expect("hash"),
            [
                0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
                0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
                0xf2, 0x00, 0x15, 0xad,
            ]
        );
    }

    #[test]
    fn constant_time_eq_checks_sha256_digests() {
        assert!(constant_time_eq(
            &hash_api_key("secret").expect("hash"),
            &hash_api_key("secret").expect("hash")
        ));
        assert!(!constant_time_eq(
            &hash_api_key("secret").expect("hash"),
            &hash_api_key("secrex").expect("hash")
        ));
        assert!(!constant_time_eq(
            &hash_api_key("secret").expect("hash"),
            &hash_api_key("secret-more").expect("hash")
        ));
    }
}
