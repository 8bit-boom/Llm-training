//! Regenerates the committed sample bundle: `cargo run -p ndforge-core --example make_sample`
fn main() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/sample.ndtrain");
    ndforge_core::sample::build(&path, 1, None).expect("write the sample bundle");
    println!("{}", path.display());
}
