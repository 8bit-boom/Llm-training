//! Checksums and the detached ed25519 signature.
//!
//! `SHA256SUMS.txt`: lines `<64 hex>  <file name>` (`sha256sum` output; a `*` before the name is accepted).
//! Signature file `<installer>.sig`: the 64-byte ed25519 signature as BASE64 TEXT (standard alphabet, whitespace around it ignored). It signs the message
//! `ndforge-update-v1:` followed by the 64 lowercase hex characters of the installer's SHA-256 (so a signature cannot be reused for another file or format).
use base64::Engine;
use ed25519_dalek::{Signature, VerifyingKey};

pub const SIGN_PREFIX: &str = "ndforge-update-v1:";

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The hash listed for `name`, lowercase.
pub fn sum_for(sums: &str, name: &str) -> Option<String> {
    sums.lines().find_map(|l| {
        let l = l.trim();
        let (h, rest) = l.split_once(char::is_whitespace)?;
        let file = rest.trim().trim_start_matches('*');
        let file = file.rsplit(['/', '\\']).next().unwrap_or(file);
        (file == name && h.len() == 64 && h.chars().all(|c| c.is_ascii_hexdigit())).then(|| h.to_ascii_lowercase())
    })
}

pub fn message(sha256_hex: &str) -> Vec<u8> {
    format!("{SIGN_PREFIX}{}", sha256_hex.to_ascii_lowercase()).into_bytes()
}

/// A public key from its base64 text (32 bytes).
pub fn parse_key(text: &str) -> Option<[u8; 32]> {
    let raw = base64::engine::general_purpose::STANDARD.decode(text.trim()).ok()?;
    let arr: [u8; 32] = raw.try_into().ok()?;
    VerifyingKey::from_bytes(&arr).ok().map(|_| arr)
}

/// true when the signature (base64 text) is a valid signature of this installer hash under `key`.
pub fn verify(key: &[u8; 32], sig_text: &str, sha256_hex: &str) -> bool {
    let Ok(vk) = VerifyingKey::from_bytes(key) else { return false };
    let Ok(raw) = base64::engine::general_purpose::STANDARD.decode(sig_text.trim()) else { return false };
    let Ok(sig) = Signature::from_slice(&raw) else { return false };
    vk.verify_strict(&message(sha256_hex), &sig).is_ok()
}
