//! Native Sign in with Apple login endpoint.
//!
//! The Mac App Store requires Apple sign-in to complete via the native
//! `ASAuthorizationController` sheet (App Review Guideline 4: "without leaving
//! the app"). The native flow produces an identity token directly — no
//! authorization code, no browser round-trip — so this endpoint verifies the
//! token against Apple's public keys and mints TrailBase tokens through the
//! same session path as the web OAuth callback.
//!
//! Key differences from the web flow:
//! - The token's `aud` is the App ID (`native_client_id`), not the Services ID
//!   used by the web flow — the audiences differ, hence the separate config.
//! - Replay protection is the `nonce` claim: the client sent
//!   `sha256(raw_nonce)` with the authorization request; we re-hash the raw
//!   nonce from the request body and compare.
//! - Email is only included by Apple on the FIRST authorization of the app.
//!   Repeat logins must therefore succeed without it: users are matched by
//!   Apple's team-stable `sub` and the minted auth token carries the stored
//!   email from the database.

use axum::extract::{Json, State};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::AppState;
use crate::auth::AuthError;
use crate::auth::oauth::OAuthUser;
use crate::auth::oauth::providers::apple::{
  decode_id_token_with_keys, extract_kid, fetch_apple_public_keys, verify_nonce_claim,
};
use crate::auth::oauth::users::{create_user_for_external_provider, user_by_provider_id};
use crate::auth::tokens::{FreshTokens, mint_new_tokens};
use crate::auth::user::DbUser;
use crate::config::proto;

/// Apple's provider name as configured in `auth.oauth_providers`.
const APPLE_PROVIDER_NAME: &str = "apple";

#[derive(Debug, Deserialize, ToSchema)]
pub struct AppleNativeLoginRequest {
  /// The identity token from `ASAuthorizationAppleIDCredential.identityToken`.
  pub identity_token: String,
  /// The raw client-generated nonce whose SHA-256 hash was sent with the
  /// authorization request (`ASAuthorizationOpenIDRequest.nonce`).
  pub nonce: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct AppleNativeTokenResponse {
  pub auth_token: String,
  pub refresh_token: String,
  pub csrf_token: String,
}

/// Logs users in with a native Sign in with Apple identity token.
#[utoipa::path(
  post,
  path = "/oauth/apple/native",
  tag = "auth",
  request_body = AppleNativeLoginRequest,
  responses(
    (status = 200, description = "Converts a verified identity token to auth tokens.", body = AppleNativeTokenResponse),
    (status = 400, description = "Malformed token or nonce mismatch."),
    (status = 401, description = "Token failed signature/issuer/audience/expiry verification."),
    (status = 424, description = "First-time login without a (verified) email claim.")
  )
)]
pub(crate) async fn native_apple_login_handler(
  State(state): State<AppState>,
  Json(request): Json<AppleNativeLoginRequest>,
) -> Result<Json<AppleNativeTokenResponse>, AuthError> {
  let auth_options = state.auth_options();
  let Some(oauth_entry) = auth_options.lookup_oauth_provider(APPLE_PROVIDER_NAME) else {
    return Err(AuthError::OAuthProviderNotFound);
  };

  // Fail closed when the native audience isn't configured (see
  // `OAuthProviderConfig.native_client_id`).
  let native_client_id = oauth_entry
    .provider
    .native_client_id()
    .ok_or(AuthError::BadRequest(
      "native sign-in is not configured for this provider",
    ))?
    .to_string();

  // Structural pre-parse: rejects malformed tokens locally, before any
  // outbound request towards Apple's keys endpoint.
  extract_kid(&request.identity_token)?;

  let http_client = reqwest::ClientBuilder::new()
    // Following redirects might set us up for server-side request forgery (SSRF).
    .redirect(reqwest::redirect::Policy::none())
    .build()
    .map_err(|err| AuthError::Internal(err.into()))?;

  let public_keys = fetch_apple_public_keys(&http_client).await?;
  let claims = decode_id_token_with_keys(&public_keys, &request.identity_token, &native_client_id)
    .map_err(|err| {
      // Signature, kid, issuer, audience and expiry failures all mean the
      // presented token did not authenticate; details are logged inside.
      let _ = err;
      return AuthError::Unauthorized;
    })?;
  verify_nonce_claim(claims.nonce.as_deref(), &request.nonce)?;

  let oauth_user = OAuthUser {
    provider_user_id: claims.sub,
    provider_id: proto::OAuthProviderId::Apple,
    email: claims.email,
    username: None,
    verified: claims.email_verified.is_some_and(|v| v.value()),
    avatar: None,
  };

  let db_user = get_or_create_native_user(&state, oauth_user).await?;

  let (auth_token_ttl, refresh_token_ttl) = state.access_config(|c| c.auth.token_ttls());
  let FreshTokens {
    auth_token_claims,
    refresh_token,
    ..
  } = mint_new_tokens(
    state.session_conn(),
    &db_user,
    &auth_token_ttl,
    &refresh_token_ttl,
  )
  .await?;

  let auth_token = state
    .jwt()
    .encode(&auth_token_claims)
    .map_err(|err| AuthError::Internal(err.into()))?;

  return Ok(Json(AppleNativeTokenResponse {
    auth_token,
    refresh_token,
    csrf_token: auth_token_claims.csrf_token,
  }));
}

/// Matches an Apple user by the team-stable `sub` and creates the account on
/// first login. Email is only present in first-authorization tokens, so it is
/// required (and must be verified) exactly when the user is new.
async fn get_or_create_native_user(
  state: &AppState,
  oauth_user: OAuthUser,
) -> Result<DbUser, AuthError> {
  if let Some(existing_user) = user_by_provider_id(
    state.user_conn(),
    proto::OAuthProviderId::Apple,
    oauth_user.provider_user_id.clone(),
  )
  .await?
  {
    return Ok(existing_user);
  }

  if oauth_user.email.is_none() {
    return Err(AuthError::FailedDependency(
      "missing email address: retry after revoking the app in Apple ID settings".into(),
    ));
  }
  if !oauth_user.verified {
    return Err(AuthError::FailedDependency(
      "email address not verified".into(),
    ));
  }

  let user_identifier = state
    .access_config(|c| c.auth.user_identifier)
    .and_then(|ui| ui.try_into().ok())
    .unwrap_or(proto::UserIdentifier::Undefined);

  let db_user =
    create_user_for_external_provider(state.user_conn(), user_identifier, oauth_user).await?;

  // This should never happen. We only ever create a new local user here for verified users.
  if !db_user.unverified_email.is_none() {
    return Err(AuthError::Internal(
      "OAuth users are expected to be verified".into(),
    ));
  }

  return Ok(db_user);
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::app_state::test_state;

  fn apple_user(email: Option<String>, verified: bool) -> OAuthUser {
    return OAuthUser {
      provider_user_id: format!(
        "apple-sub-{}",
        crate::rand::random_numeric_and_lowercase(10)
      ),
      provider_id: proto::OAuthProviderId::Apple,
      email,
      username: None,
      verified,
      avatar: None,
    };
  }

  /// Known-answer test pinning the cross-platform nonce contract: the claim
  /// carries the lowercase-hex SHA-256 of the raw nonce string. The same
  /// vector is referenced in the macOS (Rust) and iOS (Swift) clients.
  #[test]
  fn nonce_claim_matches_sha256_of_test_vector() {
    // sha256("test") in lowercase hex.
    const EXPECTED: &str = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";

    assert!(verify_nonce_claim(Some(EXPECTED), "test").is_ok());
  }

  #[test]
  fn nonce_claim_rejects_mismatch_and_absence() {
    assert!(verify_nonce_claim(Some("deadbeef"), "test").is_err());
    assert!(verify_nonce_claim(None, "test").is_err());
  }

  #[tokio::test]
  async fn existing_apple_user_logs_in_without_email_claim() {
    let state = test_state(None).await.unwrap();

    let created = get_or_create_native_user(
      &state,
      apple_user(Some("first@privaterelay.appleid.com".to_string()), true),
    )
    .await
    .unwrap();

    // Repeat login: Apple sends no email claim for known users.
    let repeat = OAuthUser {
      provider_user_id: created.provider_user_id.clone().unwrap(),
      provider_id: proto::OAuthProviderId::Apple,
      email: None,
      username: None,
      verified: false,
      avatar: None,
    };
    let user = get_or_create_native_user(&state, repeat).await.unwrap();

    assert_eq!(user.id, created.id);
    assert_eq!(
      user.email,
      Some("first@privaterelay.appleid.com".to_string())
    );
  }

  #[tokio::test]
  async fn new_user_without_email_is_rejected() {
    let state = test_state(None).await.unwrap();

    let result = get_or_create_native_user(&state, apple_user(None, false)).await;

    assert!(matches!(result, Err(AuthError::FailedDependency(_))));
  }

  #[tokio::test]
  async fn new_user_with_unverified_email_is_rejected() {
    let state = test_state(None).await.unwrap();

    let result = get_or_create_native_user(
      &state,
      apple_user(Some("unverified@example.com".to_string()), false),
    )
    .await;

    assert!(matches!(result, Err(AuthError::FailedDependency(_))));
  }
}
