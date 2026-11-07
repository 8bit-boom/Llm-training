//! Teacher-generated training examples (step 2, "Better examples"): a bigger model running on this PC or the TrueNAS box writes deeper
//! questions, rules practice, NPC role-play, history questions, lore-search questions with context, "not in the lore" answers, a little
//! general chat and house-style answers; a second pass then checks every answer against its source and drops the unsupported ones.
pub mod client;
pub mod gen;

pub use client::{extract_json, strip_thinking, Teacher, TeacherConfig};
pub use gen::{run, Options, KINDS};
