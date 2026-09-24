//! Labelme primitives, embedded images, bitmap masks, and grouped pose points.
//! Pose grouping requires Category.keypoints (or the exported categories sidecar).
use super::{Backend, ExportOptions, ExportReport, Frontend, Identity, read_json, write_json};
use crate::{
    Annotation, Category, Dataset, Keypoint, Mask, Metadata, Point, Raster, Rect, Result, Sample,
    Shape, Visibility, invalid,
};
use base64::{Engine, engine::general_purpose::STANDARD};
use image::ImageEncoder;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

pub struct Reader {
    pub directory: PathBuf,
    pub categories: Option<Vec<Category>>,
}
impl Frontend for Reader {
    fn read(&self) -> Result<Dataset> {
        read(&self.directory, self.categories.clone())
    }
}
pub struct Writer {
    pub directory: PathBuf,
    pub options: ExportOptions,
}
impl Backend for Writer {
    fn write(&self, dataset: &Dataset) -> Result<ExportReport> {
        write(dataset, &self.directory, self.options)
    }
}
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Document {
    #[serde(default = "version")]
    version: String,
    #[serde(default)]
    flags: Metadata,
    shapes: Vec<LabelShape>,
    #[serde(default)]
    image_path: Option<String>,
    #[serde(default)]
    image_data: Option<String>,
    image_height: u32,
    image_width: u32,
    #[serde(default, rename = "_cv_ir", skip_serializing_if = "Option::is_none")]
    identity: Option<Identity>,
    #[serde(flatten)]
    metadata: Metadata,
}
fn version() -> String {
    "5.7.0".into()
}
#[derive(Serialize, Deserialize)]
struct LabelShape {
    label: String,
    points: Vec<[f32; 2]>,
    #[serde(default)]
    group_id: Option<serde_json::Value>,
    #[serde(default)]
    shape_type: Option<String>,
    #[serde(default)]
    flags: Metadata,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    mask: Option<String>,
    #[serde(flatten)]
    metadata: Metadata,
}
fn group_key(group: &Option<serde_json::Value>) -> Option<String> {
    group
        .as_ref()
        .filter(|v| !v.is_null())
        .map(ToString::to_string)
}

pub fn read(root: impl AsRef<Path>, categories: Option<Vec<Category>>) -> Result<Dataset> {
    let root = root.as_ref();
    let categories_path = root.join("_categories.json");
    let supplied = if let Some(c) = categories {
        Some(c)
    } else if categories_path.is_file() {
        Some(read_json(&categories_path)?)
    } else {
        None
    };
    let paths = super::collect_files(root, "json")?
        .into_iter()
        .filter(|p| {
            !p.file_name()
                .is_some_and(|n| n == "_categories.json" || n == "_metadata.json")
        })
        .collect::<Vec<_>>();
    let docs: Result<Vec<_>> = paths.par_iter().map(read_json::<Document>).collect();
    let docs = docs?;
    let mut categories = supplied.unwrap_or_default();
    let mut labels: HashMap<String, u32> = categories
        .iter()
        .enumerate()
        .map(|(i, c)| (c.name.clone(), i as u32))
        .collect();
    // With a schema, grouped point labels are keypoint names, not object classes.
    for doc in &docs {
        let object_groups: HashMap<_, _> = doc
            .shapes
            .iter()
            .filter(|s| s.shape_type.as_deref() != Some("point"))
            .filter_map(|s| group_key(&s.group_id).map(|g| (g, s.label.as_str())))
            .collect();
        for s in &doc.shapes {
            let pose_point = s.shape_type.as_deref() == Some("point")
                && group_key(&s.group_id)
                    .and_then(|g| object_groups.get(&g).copied())
                    .and_then(|name| labels.get(name))
                    .is_some_and(|&i| categories[i as usize].keypoints.contains(&s.label));
            if !pose_point && !labels.contains_key(&s.label) {
                let i = categories.len() as u32;
                labels.insert(s.label.clone(), i);
                categories.push(Category::new(&s.label));
            }
        }
    }
    let samples: Result<Vec<_>> = paths
        .par_iter()
        .zip(docs.into_par_iter())
        .map(|(path, doc)| {
            let image = if let Some(data) = &doc.image_data {
                Raster::decode(&STANDARD.decode(data).map_err(|e| invalid(e.to_string()))?)?
            } else {
                let name = doc
                    .image_path
                    .as_deref()
                    .ok_or_else(|| invalid("Labelme needs imageData or imagePath"))?;
                Raster::open(super::image_path(path.parent().unwrap_or(root), name)?)?
            };
            if (image.width(), image.height()) != (doc.image_width, doc.image_height) {
                return Err(invalid("Labelme dimensions disagree with image"));
            }
            let relative = path.strip_prefix(root).unwrap_or(path);
            let mut sample = Sample::new(relative.to_string_lossy(), image);
            sample.split = super::infer_split(relative);
            sample.metadata = doc.metadata;
            sample
                .metadata
                .insert("flags".into(), serde_json::to_value(doc.flags)?);
            if let Some(identity) = doc.identity {
                identity.restore(&mut sample);
            }
            let mut annotations = Vec::<Annotation>::new();
            let mut groups = HashMap::<String, usize>::new();
            let mut pending = vec![];
            for (shape_index, s) in doc.shapes.into_iter().enumerate() {
                let kind = s.shape_type.as_deref().unwrap_or("polygon");
                if kind == "point" && group_key(&s.group_id).is_some() {
                    pending.push(s);
                    continue;
                }
                let class_id = *labels
                    .get(&s.label)
                    .ok_or_else(|| invalid("unknown Labelme class"))?;
                let mut a = parse_shape(&s, class_id, doc.image_width, doc.image_height)?;
                a.id = shape_index as u64;
                if let Some(g) = group_key(&s.group_id) {
                    if let Some(&i) = groups.get(&g) {
                        let prev = &mut annotations[i];
                        if prev.class_id != a.class_id {
                            return Err(invalid("Labelme group has conflicting class labels"));
                        }
                        match (&mut prev.shape, &a.shape) {
                            (Shape::Polygons(parts), Shape::Polygons(extra)) => {
                                parts.extend(extra.clone());
                                prev.bbox = prev.shape.bounds();
                            }
                            _ => {
                                return Err(invalid(
                                    "Labelme group with multiple non-polygon shapes is ambiguous",
                                ));
                            }
                        }
                        continue;
                    }
                    groups.insert(g, annotations.len());
                }
                annotations.push(a);
            }
            let mut seen_points = std::collections::HashSet::new();
            for s in pending {
                let g = group_key(&s.group_id).expect("grouped point");
                if let Some(&i) = groups.get(&g) {
                    let a = &mut annotations[i];
                    let schema = &categories[a.class_id as usize].keypoints;
                    let k = schema.iter().position(|n| n == &s.label).ok_or_else(|| {
                        invalid(format!(
                            "grouped point '{}' requires a matching category keypoint schema",
                            s.label
                        ))
                    })?;
                    if s.points.len() != 1 || !seen_points.insert((i, k)) {
                        return Err(invalid("invalid or duplicate grouped keypoint"));
                    }
                    if a.keypoints.is_empty() {
                        a.keypoints =
                            vec![Keypoint::new(0.0, 0.0, Visibility::Absent); schema.len()];
                    }
                    let visibility =
                        if s.flags.get("occluded").and_then(|v| v.as_bool()) == Some(true) {
                            Visibility::Occluded
                        } else {
                            Visibility::Visible
                        };
                    a.keypoints[k] = Keypoint::new(s.points[0][0], s.points[0][1], visibility);
                } else {
                    annotations.push(parse_shape(
                        &s,
                        labels[&s.label],
                        doc.image_width,
                        doc.image_height,
                    )?);
                }
            }
            for mut a in annotations {
                if a.keypoints.is_empty() && !categories[a.class_id as usize].keypoints.is_empty() {
                    a.keypoints = vec![
                        Keypoint::new(0.0, 0.0, Visibility::Absent);
                        categories[a.class_id as usize].keypoints.len()
                    ];
                }
                sample.objects.push(a)?;
            }
            Ok(sample)
        })
        .collect();
    let metadata = if root.join("_metadata.json").is_file() {
        read_json(root.join("_metadata.json"))?
    } else {
        Metadata::new()
    };
    let dataset = Dataset {
        categories,
        samples: samples?,
        metadata,
    };
    dataset.validate()?;
    Ok(dataset)
}
fn parse_shape(s: &LabelShape, class_id: u32, width: u32, height: u32) -> Result<Annotation> {
    let points: Vec<_> = s.points.iter().map(|p| Point::new(p[0], p[1])).collect();
    if points.iter().any(|p| !p.is_finite()) {
        return Err(invalid("non-finite Labelme coordinates"));
    }
    let mut mask = None;
    let shape = match s.shape_type.as_deref().unwrap_or("polygon") {
        "polygon" => Shape::Polygons(vec![points]),
        "rectangle" if points.len() == 2 => Shape::Rect(Rect::from_points(&points)),
        "circle" if points.len() == 2 => Shape::Circle {
            center: points[0],
            radius: (points[0].x - points[1].x).hypot(points[0].y - points[1].y),
        },
        "point" if points.len() == 1 => Shape::Point(points[0]),
        "mask" if points.len() == 2 => {
            let encoded = s
                .mask
                .as_ref()
                .ok_or_else(|| invalid("Labelme mask has no bitmap"))?;
            let bytes = STANDARD
                .decode(encoded)
                .map_err(|e| invalid(e.to_string()))?;
            let roi = image::load_from_memory(&bytes)?.into_luma8();
            // Match Labelme's reference consumer: ties-to-even placement; fractional
            // coordinates use the bitmap extent, integer endpoints are inclusive.
            let x = points[0].x.round_ties_even() as i64;
            let y = points[0].y.round_ties_even() as i64;
            let integral = points
                .iter()
                .all(|p| p.x.fract() == 0.0 && p.y.fract() == 0.0);
            let (patch_width, patch_height) = if integral {
                let w = points[1].x as f64 - points[0].x as f64 + 1.0;
                let h = points[1].y as f64 - points[0].y as f64 + 1.0;
                if w <= 0.0 || h <= 0.0 || w > roi.width() as f64 || h > roi.height() as f64 {
                    return Err(invalid("Labelme mask extent exceeds bitmap"));
                }
                (w as i64, h as i64)
            } else {
                (roi.width() as i64, roi.height() as i64)
            };
            if x >= width as i64 || y >= height as i64 || x <= -patch_width || y <= -patch_height {
                return Err(invalid("Labelme mask has no image overlap"));
            }
            let m = Mask::from_fn(width, height, |px, py| {
                let sx = px as i64 - x;
                let sy = py as i64 - y;
                sx >= 0
                    && sy >= 0
                    && sx < patch_width
                    && sy < patch_height
                    && roi.get_pixel(sx as u32, sy as u32)[0] != 0
            })?;
            let b = m.bounds().ok_or_else(|| invalid("empty Labelme mask"))?;
            mask = Some(m);
            Shape::Rect(b)
        }
        kind => {
            return Err(crate::Error::Unsupported(format!(
                "Labelme shape '{kind}' with {} points",
                points.len()
            )));
        }
    };
    let mut a = Annotation::new(class_id, shape);
    a.mask = mask;
    a.metadata = s.metadata.clone();
    a.metadata
        .insert("flags".into(), serde_json::to_value(&s.flags)?);
    if let Some(g) = &s.group_id {
        a.metadata.insert("group_id".into(), g.clone());
    }
    Ok(a)
}
pub fn write(
    dataset: &Dataset,
    root: impl AsRef<Path>,
    options: ExportOptions,
) -> Result<ExportReport> {
    dataset.validate()?;
    let root = root.as_ref();
    let mut report = ExportReport::default();
    let mut jobs = vec![];
    for s in &dataset.samples {
        let split = super::split_name(s, options.splits)?;
        let mut shapes = vec![];
        for (i, o) in s.objects.iter().enumerate() {
            if o.is_crowd() {
                report.loss(
                    options.loss,
                    format!("{}: Labelme has no standard crowd annotation", s.uid),
                )?;
            }
            let category = &dataset.categories[o.class_id() as usize];
            let mut metadata = o.metadata().clone();
            let flags: Metadata = metadata
                .remove("flags")
                .map(serde_json::from_value)
                .transpose()?
                .unwrap_or_default();
            metadata.remove("group_id");
            super::check_metadata(&metadata, &["label", "points", "shape_type", "mask"])?;
            let make = |kind: &str, points: Vec<Point>, mask: Option<String>| LabelShape {
                label: category.name.clone(),
                points: points.iter().map(|p| [p.x, p.y]).collect(),
                group_id: Some(i.into()),
                shape_type: Some(kind.into()),
                flags: flags.clone(),
                mask,
                metadata: metadata.clone(),
            };
            if let Some(mask) = o.mask() {
                let dense: Vec<u8> = mask.to_dense().into_iter().map(|b| b * 255).collect();
                let mut bytes = vec![];
                image::codecs::png::PngEncoder::new(&mut bytes).write_image(
                    &dense,
                    mask.width(),
                    mask.height(),
                    image::ExtendedColorType::L8,
                )?;
                shapes.push(make(
                    "mask",
                    vec![
                        Point::new(0.0, 0.0),
                        Point::new((mask.width() - 1) as f32, (mask.height() - 1) as f32),
                    ],
                    Some(STANDARD.encode(bytes)),
                ));
            } else {
                match o.shape() {
                    Shape::Rect(r) => shapes.push(make(
                        "rectangle",
                        vec![Point::new(r.x, r.y), Point::new(r.right(), r.bottom())],
                        None,
                    )),
                    Shape::Circle { center, radius } => shapes.push(make(
                        "circle",
                        vec![center, Point::new(center.x + radius, center.y)],
                        None,
                    )),
                    Shape::Point(p) => {
                        let mut s = make("point", vec![p], None);
                        s.group_id = None;
                        shapes.push(s);
                    }
                    Shape::Polygons(parts) => {
                        for p in parts {
                            shapes.push(make("polygon", p, None));
                        }
                    }
                }
            }
            for (k, keypoint) in o.keypoints().iter().enumerate() {
                if keypoint.visibility == Visibility::Absent {
                    continue;
                }
                let mut flags = Metadata::new();
                if keypoint.visibility == Visibility::Occluded {
                    flags.insert("occluded".into(), true.into());
                }
                shapes.push(LabelShape {
                    label: category.keypoints[k].clone(),
                    points: vec![[keypoint.point.x, keypoint.point.y]],
                    group_id: Some(i.into()),
                    shape_type: Some("point".into()),
                    flags,
                    mask: None,
                    metadata: Metadata::new(),
                });
            }
            report.annotations += 1;
        }
        let doc = Document {
            version: version(),
            flags: s
                .metadata
                .get("flags")
                .cloned()
                .map(serde_json::from_value)
                .transpose()?
                .unwrap_or_default(),
            shapes,
            image_path: Some(format!("{}.png", s.uid)),
            image_data: None,
            image_height: s.image.height(),
            image_width: s.image.width(),
            identity: Some(Identity::of(s, options.splits)),
            metadata: Metadata::new(),
        };
        jobs.push((s, split, doc));
        report.samples += 1;
    }
    super::prepare_output(root)?;
    for (_, split, _) in &jobs {
        std::fs::create_dir_all(root.join(split))?;
    }
    jobs.par_iter().try_for_each(|(s, split, doc)| {
        s.image
            .write_png(root.join(split).join(format!("{}.png", s.uid)))?;
        write_json(root.join(split).join(format!("{}.json", s.uid)), doc)
    })?;
    write_json(root.join("_categories.json"), &dataset.categories)?;
    write_json(root.join("_metadata.json"), &dataset.metadata)?;
    Ok(report)
}
