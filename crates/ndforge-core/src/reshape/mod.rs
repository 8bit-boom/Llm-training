//! Slim or grow a model: cut whole blocks of layers out of a model (prune) or copy a span of layers in (depth up-scaling), as a new base model that
//! the rest of the program (training, testing, export) uses like any other. `plan` is the pure arithmetic (sizes, limits, checks); `runs` runs the
//! worker jobs and keeps the result under `models/<name>/` in the project. The method and its limits are written up in docs/RESHAPE.md.
pub mod auto;
pub mod auto_fx;
pub mod auto_plan;
pub mod auto_progress;
pub mod auto_report;
pub mod auto_change;
pub mod auto_speed;
pub mod auto_time;
pub mod heal_bench;
pub mod heal_speed;
pub mod heal_rounds;
pub mod heal_source;
pub mod plan;
pub mod runs;
pub mod search;
pub mod thinkcheck;
pub mod tracegen;
