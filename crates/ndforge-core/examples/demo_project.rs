//! Makes a project with the sample world imported, for trying the window: `cargo run -p ndforge-core --example demo_project -- <projects dir> [dataset]` (with `dataset`, also makes and saves a training-data version)
use ndforge_core::{bundle::open_bundle, project::Project, sample, settings::Settings};

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("a projects folder"));
    let mut p = Project::create(&dir, "Ashfall Reach").expect("create");
    let f = dir.join("sample.ndtrain");
    sample::build(&f, 1, None).expect("sample");
    p.import_bundle(&open_bundle(&f).expect("open")).expect("import");
    if std::env::args().nth(2).as_deref() == Some("dataset") {
        use ndforge_core::dataset::{builders::Recipe, store};
        store::generate_draft(&p, &Recipe::default()).expect("draft");
        store::save_version(&p).expect("version");
    }
    let mut s = Settings::load();
    s.remember_project(&p.folder);
    s.save();
    println!("{}", p.folder.display());
}
