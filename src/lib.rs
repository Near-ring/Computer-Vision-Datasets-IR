//! Data-oriented computer-vision datasets: format adapters, native IR, and parallel passes.
//!
//! All geometry uses absolute pixel-edge coordinates; images use packed row-major RGB8.
//! See the crate README for format fidelity and coordinate conventions.
#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

pub mod formats;
pub mod geometry;
pub mod ir;
pub mod packed;
pub mod transforms;

pub use geometry::{Point, Rect, Shape};
pub use ir::{
    Annotation, Category, Dataset, DatasetIndex, Keypoint, Mask, Metadata, ObjectRef, ObjectTable,
    Provenance, Raster, Sample, Split, Visibility,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml_ng::Error),
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("invalid dataset: {0}")]
    Invalid(String),
    #[error("unsupported conversion: {0}")]
    Unsupported(String),
}
pub type Result<T> = std::result::Result<T, Error>;
pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

pub use uuid::Uuid;
