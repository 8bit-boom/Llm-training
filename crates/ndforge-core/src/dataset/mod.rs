//! Making training data from a world (step 2): clean the text (`markup`), build readable articles (`docs`), split long ones (`chunk`),
//! generate examples (`builders`), keep a reviewable draft (`store`) and save frozen versions.
pub mod markup;
pub mod chunk;
pub mod docs;
pub mod builders;
pub mod split;
pub mod store;
