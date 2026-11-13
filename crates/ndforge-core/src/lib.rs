//! ND Lore Forge core: everything that is not the window. Training bundles (`bundle`, `diff`), projects (`project`), app settings
//! (`settings`) and the instruction texts (`help`). No GUI or GPU code lives here, so it builds and tests anywhere.

pub mod bundle;
pub mod dataset;
pub mod diagnose;
pub mod diff;
pub mod envmgr;
pub mod evaluate;
pub mod error;
pub mod export;
pub mod help;
pub mod model;
pub mod paths;
pub mod power;
pub mod project;
pub mod secrets;
pub mod remote;
pub mod reshape;
pub mod train;
pub mod sample;
pub mod teacher;
pub mod unsloth;
pub mod settings;

pub use error::{Error, Result};
