//! Training in Unsloth Desktop: the client, the run and the stand-in Unsloth they are tested against.
mod common;

use common::fake_unsloth::{Fake, KEY};
use tempfile::tempdir;

#[test]
fn the_stand_in_answers_health_with_and_without_a_key() {
    let d = tempdir().unwrap();
    let f = Fake::start(d.path());
    let plain: serde_json::Value = ureq::get(&format!("{}/api/health", f.url)).call().unwrap().into_json().unwrap();
    assert_eq!(plain["service"], "Unsloth UI Backend");
    assert!(plain["version"].is_null());
    let keyed: serde_json::Value = ureq::get(&format!("{}/api/health", f.url)).set("Authorization", &format!("Bearer {KEY}")).call().unwrap().into_json().unwrap();
    assert!(keyed["version"].is_string());
    assert!(matches!(ureq::get(&format!("{}/v1/models", f.url)).call(), Err(ureq::Error::Status(401, _))));
}
