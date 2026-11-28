//! The PUBLIC half of the key that signs the installers (docs/UPDATES.md). Empty until the owner has made a key pair with `scripts/make_update_key.py`
//! and pasted the printed public key here: 32 raw bytes, standard base64 (44 characters). While it is empty the program runs in "checksum only" mode and says so.
//! Never put the private key anywhere in the repository.
pub const UPDATE_PUBLIC_KEY: &str = "";
