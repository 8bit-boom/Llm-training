//! Slim or grow a model: cut whole blocks of layers out of a model (prune) or copy a span of layers in (depth up-scaling), as a new base model that
//! the rest of the program (training, testing, export) uses like any other. `plan` is the pure arithmetic (sizes, limits, checks); `runs` runs the
//! worker jobs and keeps the result under `models/<name>/` in the project. The method and its limits are written up in docs/RESHAPE.md.
pub mod plan;
pub mod runs;
