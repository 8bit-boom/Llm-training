//! Training on another computer: the TrueNAS server's graphics card. `server` runs there (in a container, `crates/ndforge-server`), `client` and
//! `run` are what this program uses: send the data, train, show the same live events as a local run and bring the trained adapter back.
//!
//! Protocol (HTTP + JSON, every request carries `Authorization: Bearer <token>`):
//! ```text
//! GET    /v1/status                         what the server has: card, busy, Ollama
//! PUT    /v1/jobs/<id>/files/<name>         train.jsonl or init_adapter.zip
//! POST   /v1/jobs/<id>                      {"job": <worker job>}  start (409 when busy)
//! GET    /v1/jobs/<id>                      state
//! GET    /v1/jobs/<id>/events?from=<n>      {"events": [...], "next": n, "state": "running|done|stopped|failed"}
//! POST   /v1/jobs/<id>/stop                 graceful stop (the worker saves a checkpoint)
//! GET    /v1/jobs/<id>/adapter.zip          the trained adapter
//! DELETE /v1/jobs/<id>                      forget a finished job
//! POST   /v1/ollama/unload                  free the card: unload every model Ollama has loaded
//! ```
pub mod client;
pub mod run;
pub mod server;
pub mod zipio;
