//! One error type whose text is written for the user (it is shown in the window as it is).
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

/// The text of the error a stopped job ends with. The window and several callers recognise it; use `Error::cancelled()` / `is_cancelled()`.
pub const CANCELLED: &str = "Cancelled.";

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn new(msg: impl Into<String>) -> Self {
        Error(msg.into())
    }

    /// The error of a job that was stopped on request (text: "Cancelled.").
    pub fn cancelled() -> Self {
        Error(CANCELLED.to_string())
    }

    pub fn is_cancelled(&self) -> bool {
        self.0 == CANCELLED
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error(format!("File problem: {e}"))
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error(format!("Not valid JSON: {e}"))
    }
}
