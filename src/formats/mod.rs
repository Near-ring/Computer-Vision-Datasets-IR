//! Explicit format adapters. Unsupported annotation conversions return errors by default.
pub mod coco;
pub mod labelme;
pub mod yolo;

use crate::{Dataset, Metadata, Result, Sample, Split, Uuid, invalid};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::{Component, Path, PathBuf},
};

pub trait Frontend {
    fn read(&self) -> Result<Dataset>;
}
pub trait Backend {
    fn write(&self, dataset: &Dataset) -> Result<ExportReport>;
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SplitPolicy {
    #[default]
    Preserve,
    Ignore,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LossPolicy {
    #[default]
    Error,
    Allow,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct ExportOptions {
    pub splits: SplitPolicy,
    pub loss: LossPolicy,
}
#[derive(Clone, Debug, Default)]
pub struct ExportReport {
    pub samples: usize,
    pub annotations: usize,
    pub warnings: Vec<String>,
}
impl ExportReport {
    pub(crate) fn loss(&mut self, policy: LossPolicy, message: impl Into<String>) -> Result<()> {
        let message = message.into();
        if policy == LossPolicy::Error {
            return Err(crate::Error::Unsupported(message));
        }
        self.warnings.push(message);
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Identity {
    #[serde(default)]
    pub provenance: Vec<crate::ir::Provenance>,
    pub uid: Uuid,
    pub id: String,
    pub split: Split,
    pub metadata: Metadata,
}
impl Identity {
    pub fn of(s: &Sample, policy: SplitPolicy) -> Self {
        Self {
            uid: s.uid,
            id: s.id.clone(),
            split: if policy == SplitPolicy::Preserve {
                s.split.clone()
            } else {
                Split::Unassigned
            },
            metadata: s.metadata.clone(),
            provenance: s.provenance.clone(),
        }
    }
    pub fn restore(self, s: &mut Sample) {
        s.uid = self.uid;
        s.id = self.id;
        s.split = self.split;
        s.metadata = self.metadata;
        s.provenance = self.provenance;
    }
}
pub(crate) fn read_json<T: DeserializeOwned>(p: impl AsRef<Path>) -> Result<T> {
    Ok(serde_json::from_reader(BufReader::new(File::open(p)?))?)
}
pub(crate) fn write_json(p: impl AsRef<Path>, v: &impl Serialize) -> Result<()> {
    let mut w = BufWriter::new(File::create_new(p)?);
    serde_json::to_writer_pretty(&mut w, v)?;
    w.flush()?;
    Ok(())
}
pub(crate) fn infer_split(p: &Path) -> Split {
    p.components()
        .filter_map(|c| c.as_os_str().to_str())
        .find_map(|s| match s {
            "train" => Some(Split::Train),
            "val" | "valid" | "validation" => Some(Split::Val),
            "test" => Some(Split::Test),
            _ => None,
        })
        .unwrap_or_default()
}
pub(crate) fn split_name(s: &Sample, policy: SplitPolicy) -> Result<String> {
    let name = if policy == SplitPolicy::Ignore {
        "unassigned"
    } else {
        s.split.as_str()
    };
    if name.is_empty() || name == "." || name == ".." || name.contains(['/', '\\']) {
        return Err(invalid(
            "split name must be a single safe directory component",
        ));
    }
    Ok(name.to_owned())
}
/// Input paths are confined to the declared image root, including symlink resolution.
pub(crate) fn image_path(root: &Path, name: &str) -> Result<PathBuf> {
    let normalized = name.replace('\\', "/");
    let rel = Path::new(&normalized);
    if rel.is_absolute()
        || rel
            .components()
            .any(|c| !matches!(c, Component::Normal(_) | Component::CurDir))
    {
        return Err(invalid(format!(
            "image path must stay within its root: {name}"
        )));
    }
    let base = root.canonicalize()?;
    let path = base.join(rel).canonicalize()?;
    if !path.starts_with(&base) {
        return Err(invalid("image symlink escapes its root"));
    }
    Ok(path)
}
pub(crate) fn collect_files(root: &Path, extension: &str) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in walkdir::WalkDir::new(root) {
        let entry = entry.map_err(|e| invalid(e.to_string()))?;
        if entry.file_type().is_file()
            && entry
                .path()
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case(extension))
        {
            paths.push(entry.into_path());
        }
    }
    paths.sort();
    Ok(paths)
}
pub(crate) fn prepare_output(root: &Path) -> Result<()> {
    if root.exists() {
        return Err(invalid(format!(
            "output already exists: {}",
            root.display()
        )));
    }
    if let Some(p) = root.parent() {
        std::fs::create_dir_all(p)?;
    }
    std::fs::create_dir(root)?;
    Ok(())
}

pub(crate) fn check_metadata(metadata: &Metadata, reserved: &[&str]) -> Result<()> {
    if let Some(key) = reserved.iter().find(|&&key| metadata.contains_key(key)) {
        return Err(invalid(format!(
            "metadata key '{key}' conflicts with a format field"
        )));
    }
    Ok(())
}
