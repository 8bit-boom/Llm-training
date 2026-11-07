//! ND Lore Forge core: everything that is not the window. Training bundles (`bundle`, `diff`), projects (`project`), app settings
//! (`settings`) and the instruction texts (`help`). No GUI or GPU code lives here, so it builds and tests anywhere.

pub mod bundle;
pub mod diff;
pub mod envmgr;
pub mod error;
pub mod help;
pub mod model;
pub mod paths;
pub mod project;
pub mod sample;
pub mod settings;

pub use error::{Error, Result};
