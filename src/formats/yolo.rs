//! Ultralytics YAML + image/label trees for detection, segmentation, and pose.
//! The task is explicit: numeric row widths alone cannot reliably identify a task.
use super::{Backend, ExportOptions, ExportReport, Frontend, Identity, read_json, write_json};
use crate::{
    Annotation, Category, Dataset, Keypoint, Metadata, Point, Raster, Rect, Result, Sample, Shape,
    Split, Visibility, invalid,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashSet},
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Component, Path, PathBuf},
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    Detection,
    Segmentation,
    Pose,
}
pub struct Reader {
    pub yaml: PathBuf,
    pub task: Task,
}
impl Frontend for Reader {
    fn read(&self) -> Result<Dataset> {
        read(&self.yaml, self.task)
    }
}
pub struct Writer {
    pub directory: PathBuf,
    pub task: Task,
    pub options: ExportOptions,
}
impl Backend for Writer {
    fn write(&self, dataset: &Dataset) -> Result<ExportReport> {
        write(dataset, &self.directory, self.task, self.options)
    }
}
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum Names {
    List(Vec<String>),
    Map(BTreeMap<u32, String>),
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum Paths {
    One(String),
    Many(Vec<String>),
}
impl Paths {
    fn values(self) -> Vec<String> {
        match self {
            Self::One(p) => vec![p],
            Self::Many(p) => p,
        }
    }
}
#[derive(Serialize, Deserialize)]
struct Config {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    path: Option<String>,
    names: Names,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    train: Option<Paths>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    val: Option<Paths>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    test: Option<Paths>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    kpt_shape: Option<[usize; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    kpt_names: Option<BTreeMap<u32, Vec<String>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    flip_idx: Option<Vec<usize>>,
    #[serde(flatten)]
    metadata: Metadata,
}
#[derive(Serialize, Deserialize)]
struct ObjectExtra {
    id: u64,
    is_crowd: bool,
    metadata: Metadata,
}
#[derive(Serialize, Deserialize)]
struct Entry {
    identity: Identity,
    objects: Vec<ObjectExtra>,
}
#[derive(Serialize, Deserialize)]
struct Manifest {
    categories: Vec<Category>,
    metadata: Metadata,
    samples: BTreeMap<String, Entry>,
}
fn resolve(base: &Path, p: &str) -> PathBuf {
    let p = Path::new(p);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        base.join(p)
    }
}
fn is_image(p: &Path) -> bool {
    p.extension().and_then(|e| e.to_str()).is_some_and(|e| {
        ["jpg", "jpeg", "png", "webp"]
            .iter()
            .any(|s| e.eq_ignore_ascii_case(s))
    })
}
fn image_files(base: &Path, path: &str) -> Result<Vec<PathBuf>> {
    let path = resolve(base, path);
    let mut out = vec![];
    if path.is_dir() {
        for e in walkdir::WalkDir::new(&path) {
            let e = e.map_err(|e| invalid(e.to_string()))?;
            if e.file_type().is_file() && is_image(e.path()) {
                out.push(e.into_path());
            }
        }
    } else if path.extension().is_some_and(|e| e == "txt") {
        for line in BufReader::new(File::open(&path)?).lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let source = if line.starts_with("./") {
                resolve(path.parent().unwrap_or(base), line)
            } else {
                resolve(base, line)
            };
            if !is_image(&source) {
                return Err(invalid("unsupported image extension in YOLO list"));
            }
            out.push(source);
        }
    } else if is_image(&path) && path.is_file() {
        out.push(path);
    } else {
        return Err(invalid(format!(
            "YOLO split path does not exist or is unsupported: {}",
            path.display()
        )));
    }
    out.sort();
    Ok(out)
}
fn label_path(image: &Path) -> Result<PathBuf> {
    let components: Vec<_> = image.components().collect();
    let index = components
        .iter()
        .rposition(|c| *c == Component::Normal("images".as_ref()))
        .ok_or_else(|| invalid("YOLO image path needs an 'images' component"))?;
    let mut path = PathBuf::new();
    for (i, c) in components.iter().enumerate() {
        if i == index {
            path.push("labels");
        } else {
            path.push(c.as_os_str());
        }
    }
    path.set_extension("txt");
    Ok(path)
}
fn normalized(v: f32) -> Result<f32> {
    if v.is_finite() && (-1e-6..=1.000001).contains(&v) {
        Ok(v.clamp(0.0, 1.0))
    } else {
        Err(invalid(
            "YOLO coordinate must be finite and normalized to [0,1]",
        ))
    }
}
pub fn read(yaml: impl AsRef<Path>, task: Task) -> Result<Dataset> {
    let yaml = yaml.as_ref();
    let config: Config = serde_yaml_ng::from_reader(BufReader::new(File::open(yaml)?))?;
    let base = resolve(
        yaml.parent().unwrap_or(Path::new(".")),
        config.path.as_deref().unwrap_or("."),
    );
    let base = base.canonicalize()?;
    let names = match config.names {
        Names::List(n) => n,
        Names::Map(n) => {
            if n.keys().enumerate().any(|(i, &k)| k as usize != i) {
                return Err(invalid("YOLO class IDs must be dense and zero-based"));
            }
            n.into_values().collect()
        }
    };
    let schema = if task == Task::Pose {
        let [n, d] = config
            .kpt_shape
            .ok_or_else(|| invalid("YOLO pose requires kpt_shape"))?;
        if n == 0 || !(2..=3).contains(&d) || n.checked_mul(d).is_none() {
            return Err(invalid("invalid YOLO keypoint shape"));
        }
        Some((n, d))
    } else {
        None
    };
    let manifest: Option<Manifest> = if base.join("_ir_manifest.json").is_file() {
        Some(read_json(base.join("_ir_manifest.json"))?)
    } else {
        None
    };
    let categories = if let Some(m) = &manifest {
        if m.categories.iter().map(|c| &c.name).ne(names.iter()) {
            return Err(invalid("YOLO manifest categories disagree with YAML"));
        }
        m.categories.clone()
    } else {
        names
            .into_iter()
            .enumerate()
            .map(|(i, name)| {
                let mut c = Category::new(name);
                if let Some((n, _)) = schema {
                    c.keypoints = config
                        .kpt_names
                        .as_ref()
                        .and_then(|names| names.get(&(i as u32)))
                        .cloned()
                        .unwrap_or_else(|| (0..n).map(|i| format!("point_{i}")).collect());
                }
                c
            })
            .collect()
    };
    if let Some((n, _)) = schema
        && categories.iter().any(|c| c.keypoints.len() != n)
    {
        return Err(invalid("YOLO keypoint names disagree with kpt_shape"));
    }
    if let Some(p) = &config.flip_idx
        && (schema.is_some_and(|(n, _)| p.len() != n)
            || p.iter().enumerate().any(|(i, &j)| p.get(j) != Some(&i)))
    {
        return Err(invalid("invalid YOLO flip_idx"));
    }
    let mut jobs = Vec::new();
    let mut seen = HashSet::new();
    for (split, paths) in [
        (Split::Train, config.train),
        (Split::Val, config.val),
        (Split::Test, config.test),
    ] {
        if let Some(paths) = paths {
            for path in paths.values() {
                for p in image_files(&base, &path)? {
                    let p = p.canonicalize()?;
                    if !seen.insert(p.clone()) {
                        return Err(invalid("image occurs more than once in YOLO splits"));
                    }
                    jobs.push((split.clone(), p));
                }
            }
        }
    }
    // Custom/unassigned splits are recorded in our sidecar but have no standard YAML key.
    if let Some(m) = &manifest {
        for key in m.samples.keys() {
            let p = super::image_path(&base, key)?;
            if seen.insert(p.clone()) {
                jobs.push((Split::Unassigned, p));
            }
        }
    }
    let samples: Result<Vec<_>> = jobs
        .par_iter()
        .map(|(split, path)| {
            let image = Raster::open(path)?;
            let w = image.width() as f32;
            let h = image.height() as f32;
            let key = path
                .strip_prefix(&base)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/");
            let mut sample = Sample::new(&key, image);
            sample.split = split.clone();
            let label = label_path(path)?;
            if label.is_file() {
                for (line_number, line) in BufReader::new(File::open(&label)?).lines().enumerate() {
                    let line = line?;
                    let tokens: Vec<_> = line.split_whitespace().collect();
                    if tokens.is_empty() {
                        continue;
                    }
                    let class = tokens[0].parse::<u32>().map_err(|_| {
                        invalid(format!(
                            "{}:{}: invalid class ID",
                            label.display(),
                            line_number + 1
                        ))
                    })?;
                    if class as usize >= categories.len() {
                        return Err(invalid("YOLO class ID out of range"));
                    }
                    let values: Result<Vec<f32>> = tokens[1..]
                        .iter()
                        .map(|t| {
                            t.parse::<f32>()
                                .map_err(|_| invalid("invalid YOLO numeric value"))
                        })
                        .collect();
                    let v = values?;
                    let mut a = match task {
                        Task::Segmentation => {
                            if v.len() < 6 || v.len() % 2 != 0 {
                                return Err(invalid(
                                    "YOLO segmentation requires at least three coordinate pairs",
                                ));
                            }
                            let points: Result<Vec<_>> = v
                                .as_chunks::<2>()
                                .0
                                .iter()
                                .map(|p| {
                                    Ok(Point::new(normalized(p[0])? * w, normalized(p[1])? * h))
                                })
                                .collect();
                            Annotation::new(class, Shape::Polygons(vec![points?]))
                        }
                        Task::Detection | Task::Pose => {
                            let expected = if let Some((n, d)) = schema {
                                4 + n * d
                            } else {
                                4
                            };
                            if v.len() != expected {
                                return Err(invalid(format!(
                                    "YOLO row expects {expected} values, got {}",
                                    v.len()
                                )));
                            }
                            let (cx, cy, bw, bh) = (
                                normalized(v[0])? * w,
                                normalized(v[1])? * h,
                                normalized(v[2])? * w,
                                normalized(v[3])? * h,
                            );
                            if bw <= 0.0 || bh <= 0.0 {
                                return Err(invalid("YOLO box must have positive size"));
                            }
                            let mut a = Annotation::new(
                                class,
                                Shape::Rect(Rect::new(cx - bw / 2.0, cy - bh / 2.0, bw, bh)),
                            );
                            if let Some((_, d)) = schema {
                                for k in v[4..].chunks_exact(d) {
                                    let visibility = if d == 3 {
                                        if !k[2].is_finite()
                                            || k[2].fract() != 0.0
                                            || !(0.0..=2.0).contains(&k[2])
                                        {
                                            return Err(invalid(
                                                "invalid YOLO keypoint visibility",
                                            ));
                                        }
                                        Visibility::try_from(k[2] as u8)?
                                    } else {
                                        Visibility::Visible
                                    };
                                    a.keypoints.push(Keypoint::new(
                                        normalized(k[0])? * w,
                                        normalized(k[1])? * h,
                                        visibility,
                                    ));
                                }
                            }
                            a
                        }
                    };
                    a.id = line_number as u64;
                    sample.objects.push(a)?;
                }
            }
            if let Some(entry) = manifest.as_ref().and_then(|m| m.samples.get(&key)) {
                entry.identity.clone().restore(&mut sample);
                if entry.objects.len() != sample.objects.len() {
                    return Err(invalid("YOLO sidecar annotation count mismatch"));
                }
                let mut objects = crate::ObjectTable::default();
                for (o, extra) in sample.objects.iter().zip(&entry.objects) {
                    let mut a = o.to_owned();
                    a.id = extra.id;
                    a.is_crowd = extra.is_crowd;
                    a.metadata = extra.metadata.clone();
                    objects.push(a)?;
                }
                sample.objects = objects;
            }
            Ok(sample)
        })
        .collect();
    let mut metadata = manifest.map(|m| m.metadata).unwrap_or_default();
    if !config.metadata.is_empty() {
        metadata.insert("yolo".into(), serde_json::to_value(config.metadata)?);
    }
    if let Some(p) = config.flip_idx {
        metadata.insert("yolo_flip_idx".into(), serde_json::to_value(p)?);
    }
    let dataset = Dataset {
        categories,
        samples: samples?,
        metadata,
    };
    dataset.validate()?;
    Ok(dataset)
}
pub fn write(
    dataset: &Dataset,
    root: impl AsRef<Path>,
    task: Task,
    options: ExportOptions,
) -> Result<ExportReport> {
    dataset.validate()?;
    let root = root.as_ref();
    let mut report = ExportReport::default();
    let n = dataset.categories.first().map_or(0, |c| c.keypoints.len());
    if task == Task::Pose && (n == 0 || dataset.categories.iter().any(|c| c.keypoints.len() != n)) {
        return Err(invalid(
            "YOLO pose requires equal nonzero keypoint counts for all classes",
        ));
    }
    let mut manifest = Manifest {
        categories: dataset.categories.clone(),
        metadata: dataset.metadata.clone(),
        samples: BTreeMap::new(),
    };
    let mut jobs = vec![];
    let mut splits = HashSet::new();
    for s in &dataset.samples {
        let split = super::split_name(s, options.splits)?;
        splits.insert(split.clone());
        let mut text = String::new();
        let w = s.image.width() as f32;
        let h = s.image.height() as f32;
        for o in s.objects.iter() {
            let mut v = vec![];
            match task {
                Task::Detection | Task::Pose => {
                    if o.mask().is_some() || !matches!(o.shape(), Shape::Rect(_)) {
                        report.loss(
                            options.loss,
                            format!("{}: geometry reduced to a YOLO bounding box", s.uid),
                        )?;
                    }
                    if task == Task::Detection && !o.keypoints().is_empty() {
                        report.loss(
                            options.loss,
                            format!("{}: YOLO detection omits keypoints", s.uid),
                        )?;
                    }
                    let b = o.bbox();
                    if b.width <= 0.0 || b.height <= 0.0 {
                        return Err(invalid("YOLO cannot export zero-area boxes"));
                    }
                    v.extend([
                        (b.x + b.width / 2.0) / w,
                        (b.y + b.height / 2.0) / h,
                        b.width / w,
                        b.height / h,
                    ]);
                    if task == Task::Pose {
                        if !o.keypoints().is_empty() && o.keypoints().len() != n {
                            return Err(invalid("pose keypoint count mismatch"));
                        }
                        if o.keypoints().is_empty() {
                            v.extend(vec![0.0; n * 3]);
                        } else {
                            for k in o.keypoints() {
                                if k.visibility == Visibility::Absent {
                                    v.extend([0.0, 0.0, 0.0]);
                                } else {
                                    v.extend([
                                        k.point.x / w,
                                        k.point.y / h,
                                        k.visibility as u8 as f32,
                                    ]);
                                }
                            }
                        }
                    }
                }
                Task::Segmentation => {
                    if !o.keypoints().is_empty() {
                        report.loss(
                            options.loss,
                            format!("{}: YOLO segmentation omits keypoints", s.uid),
                        )?;
                    }
                    let parts = if let Some(m) = o.mask() {
                        report.loss(
                            options.loss,
                            format!(
                                "{}: bitmap mask converted to polygon contours at pixel centers",
                                s.uid
                            ),
                        )?;
                        let dense = m.to_dense();
                        let image = image::GrayImage::from_raw(m.width(), m.height(), dense)
                            .ok_or_else(|| invalid("mask dimensions"))?;
                        let contours = imageproc::contours::find_contours::<i32>(&image);
                        if contours
                            .iter()
                            .any(|c| c.border_type == imageproc::contours::BorderType::Hole)
                        {
                            report.loss(
                                options.loss,
                                format!("{}: YOLO polygon cannot retain mask holes", s.uid),
                            )?;
                        }
                        contours
                            .into_iter()
                            .filter(|c| {
                                c.border_type == imageproc::contours::BorderType::Outer
                                    && c.points.len() >= 3
                            })
                            .map(|c| {
                                c.points
                                    .into_iter()
                                    .map(|p| Point::new(p.x as f32 + 0.5, p.y as f32 + 0.5))
                                    .collect()
                            })
                            .collect::<Vec<Vec<Point>>>()
                    } else {
                        let shape = o.shape();
                        if matches!(shape, Shape::Circle { .. }) {
                            report.loss(
                                options.loss,
                                format!("{}: circle approximated by 64 vertices", s.uid),
                            )?;
                        }
                        shape.polygons(64)
                    };
                    if parts.len() > 1 {
                        report.loss(
                            options.loss,
                            format!("{}: YOLO keeps only the largest polygon component", s.uid),
                        )?;
                    }
                    let part = parts
                        .into_iter()
                        .max_by(|a, b| {
                            crate::geometry::polygon_area(a)
                                .total_cmp(&crate::geometry::polygon_area(b))
                        })
                        .ok_or_else(|| invalid("object has no exportable YOLO polygon"))?;
                    v.extend(part.into_iter().flat_map(|p| [p.x / w, p.y / h]));
                }
            }
            text.push_str(&o.class_id().to_string());
            for (i, value) in v.into_iter().enumerate() {
                let visibility = task == Task::Pose && i >= 4 && (i - 4) % 3 == 2;
                let value = if visibility {
                    value
                } else {
                    normalized(value)?
                };
                text.push(' ');
                text.push_str(&value.to_string());
            }
            text.push('\n');
            report.annotations += 1;
        }
        let key = format!("images/{split}/{}.png", s.uid);
        manifest.samples.insert(
            key,
            Entry {
                identity: Identity::of(s, options.splits),
                objects: s
                    .objects
                    .iter()
                    .map(|o| ObjectExtra {
                        id: o.id(),
                        is_crowd: o.is_crowd(),
                        metadata: o.metadata().clone(),
                    })
                    .collect(),
            },
        );
        jobs.push((s, split, text));
        report.samples += 1;
    }
    let config = Config {
        path: None,
        names: Names::List(dataset.categories.iter().map(|c| c.name.clone()).collect()),
        train: splits
            .contains("train")
            .then(|| Paths::One("images/train".into())),
        val: splits
            .contains("val")
            .then(|| Paths::One("images/val".into())),
        test: splits
            .contains("test")
            .then(|| Paths::One("images/test".into())),
        kpt_shape: (task == Task::Pose).then_some([n, 3]),
        kpt_names: (task == Task::Pose).then(|| {
            dataset
                .categories
                .iter()
                .enumerate()
                .map(|(i, c)| (i as u32, c.keypoints.clone()))
                .collect()
        }),
        flip_idx: if task == Task::Pose {
            dataset
                .metadata
                .get("yolo_flip_idx")
                .cloned()
                .map(serde_json::from_value)
                .transpose()?
        } else {
            None
        },
        metadata: Metadata::new(),
    };
    if let Some(p) = &config.flip_idx
        && (p.len() != n || p.iter().enumerate().any(|(i, &j)| p.get(j) != Some(&i)))
    {
        return Err(invalid("invalid yolo_flip_idx metadata"));
    }
    super::prepare_output(root)?;
    for split in splits {
        std::fs::create_dir_all(root.join("images").join(&split))?;
        std::fs::create_dir_all(root.join("labels").join(&split))?;
    }
    jobs.par_iter().try_for_each(|(s, split, text)| {
        s.image.write_png(
            root.join("images")
                .join(split)
                .join(format!("{}.png", s.uid)),
        )?;
        let mut w = BufWriter::new(File::create_new(
            root.join("labels")
                .join(split)
                .join(format!("{}.txt", s.uid)),
        )?);
        w.write_all(text.as_bytes())?;
        w.flush()?;
        Ok::<_, crate::Error>(())
    })?;
    let mut out = BufWriter::new(File::create_new(root.join("data.yaml"))?);
    serde_yaml_ng::to_writer(&mut out, &config)?;
    out.flush()?;
    write_json(root.join("_ir_manifest.json"), &manifest)?;
    Ok(report)
}
