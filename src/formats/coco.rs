//! COCO detection, multipart polygons, compressed/uncompressed RLE, and pose.
use super::{
    Backend, ExportOptions, ExportReport, Frontend, Identity, image_path, read_json, write_json,
};
use crate::{
    Annotation, Category, Dataset, Keypoint, Mask, Metadata, Point, Raster, Rect, Result, Sample,
    Shape, Split, Visibility, invalid,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::{Path, PathBuf},
};

#[derive(Clone, Debug)]
pub struct Source {
    pub annotations: PathBuf,
    pub images: PathBuf,
    pub split: Split,
}
pub struct Reader {
    pub sources: Vec<Source>,
}
impl Frontend for Reader {
    fn read(&self) -> Result<Dataset> {
        read(&self.sources)
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
struct Document {
    images: Vec<CocoImage>,
    #[serde(default)]
    annotations: Vec<CocoAnnotation>,
    categories: Vec<CocoCategory>,
    #[serde(flatten)]
    metadata: Metadata,
}
#[derive(Serialize, Deserialize)]
struct CocoImage {
    id: u64,
    file_name: String,
    width: u32,
    height: u32,
    #[serde(default, rename = "_cv_ir", skip_serializing_if = "Option::is_none")]
    identity: Option<Identity>,
    #[serde(flatten)]
    metadata: Metadata,
}
#[derive(Serialize, Deserialize)]
struct CocoCategory {
    id: u64,
    name: String,
    #[serde(default)]
    keypoints: Vec<String>,
    #[serde(default)]
    skeleton: Vec<[u32; 2]>,
    #[serde(flatten)]
    metadata: Metadata,
}
#[derive(Serialize, Deserialize)]
struct CocoAnnotation {
    id: u64,
    image_id: u64,
    category_id: u64,
    #[serde(default)]
    bbox: Option<[f32; 4]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    segmentation: Option<Segmentation>,
    #[serde(default)]
    keypoints: Vec<f32>,
    #[serde(default)]
    iscrowd: u8,
    #[serde(default)]
    area: f64,
    #[serde(default)]
    num_keypoints: u32,
    #[serde(flatten)]
    metadata: Metadata,
}
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum Segmentation {
    Polygons(Vec<Vec<f32>>),
    Rle { size: [u32; 2], counts: Counts },
}
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum Counts {
    Plain(Vec<u32>),
    Compressed(String),
}

pub fn read(sources: &[Source]) -> Result<Dataset> {
    let mut dataset = Dataset::default();
    let mut names = HashMap::<String, u32>::new();
    let mut source_metadata = Vec::new();
    for (source_index, source) in sources.iter().enumerate() {
        let doc: Document = read_json(&source.annotations)?;
        source_metadata.push(serde_json::to_value(&doc.metadata)?);
        let mut category_map = HashMap::new();
        for mut c in doc.categories {
            c.metadata
                .entry("source_category_id".into())
                .or_insert_with(|| c.id.into());
            if c.skeleton.iter().flatten().any(|i| *i == 0) {
                return Err(invalid("COCO skeleton indices are one-based"));
            }
            let category = Category {
                name: c.name.clone(),
                keypoints: c.keypoints,
                skeleton: c
                    .skeleton
                    .into_iter()
                    .map(|[a, b]| [a - 1, b - 1])
                    .collect(),
                metadata: c.metadata,
            };
            let index = if let Some(&i) = names.get(&c.name) {
                let existing = &dataset.categories[i as usize];
                if existing.keypoints != category.keypoints
                    || existing.skeleton != category.skeleton
                {
                    return Err(invalid("inconsistent COCO keypoint schema across splits"));
                }
                i
            } else {
                let i = dataset.categories.len() as u32;
                names.insert(c.name, i);
                dataset.categories.push(category);
                i
            };
            if category_map.insert(c.id, index).is_some() {
                return Err(invalid("duplicate COCO category ID"));
            }
        }
        let mut image_ids = HashSet::new();
        for i in &doc.images {
            if !image_ids.insert(i.id) {
                return Err(invalid("duplicate COCO image ID"));
            }
        }
        let mut annotation_ids = HashSet::new();
        let mut grouped = HashMap::<u64, Vec<CocoAnnotation>>::new();
        for a in doc.annotations {
            if !image_ids.contains(&a.image_id) || !annotation_ids.insert(a.id) {
                return Err(invalid("orphan annotation or duplicate COCO annotation ID"));
            }
            grouped.entry(a.image_id).or_default().push(a);
        }
        let samples: Result<Vec<_>> = doc
            .images
            .into_par_iter()
            .map(|image| {
                let raster = Raster::open(image_path(&source.images, &image.file_name)?)?;
                if (raster.width(), raster.height()) != (image.width, image.height) {
                    return Err(invalid(format!(
                        "COCO image dimensions disagree: {}",
                        image.file_name
                    )));
                }
                let mut sample = Sample::new(format!("{source_index}/{}", image.id), raster);
                sample.split = source.split.clone();
                sample.metadata = image.metadata;
                sample
                    .metadata
                    .insert("source_file".into(), image.file_name.into());
                sample
                    .metadata
                    .insert("source_image_id".into(), image.id.into());
                if let Some(identity) = image.identity {
                    identity.restore(&mut sample);
                }
                if let Some(annotations) = grouped.get(&image.id) {
                    for a in annotations {
                        let class_id = *category_map
                            .get(&a.category_id)
                            .ok_or_else(|| invalid("unknown COCO category ID"))?;
                        let mut mask = None;
                        let bbox = a.bbox.map(|b| Rect::new(b[0], b[1], b[2], b[3]));
                        let shape = match &a.segmentation {
                            Some(Segmentation::Polygons(parts)) if !parts.is_empty() => {
                                let parts: Result<Vec<_>> = parts
                                    .iter()
                                    .map(|p| {
                                        if p.len() < 6 || p.len() % 2 != 0 {
                                            return Err(invalid("invalid COCO polygon"));
                                        }
                                        Ok(p.as_chunks::<2>()
                                            .0
                                            .iter()
                                            .map(|p| Point::new(p[0], p[1]))
                                            .collect())
                                    })
                                    .collect();
                                Shape::Polygons(parts?)
                            }
                            Some(Segmentation::Rle { size, counts }) => {
                                if *size != [image.height, image.width] {
                                    return Err(invalid("COCO RLE dimensions mismatch"));
                                }
                                let counts = match counts {
                                    Counts::Plain(c) => c.clone(),
                                    Counts::Compressed(s) => decode_counts(s)?,
                                };
                                let m = decode_mask(image.width, image.height, &counts)?;
                                let bounds = m
                                    .bounds()
                                    .or(bbox)
                                    .ok_or_else(|| invalid("empty COCO mask without bbox"))?;
                                mask = Some(m);
                                Shape::Rect(bounds)
                            }
                            _ => Shape::Rect(bbox.ok_or_else(|| {
                                invalid("COCO annotation has neither bbox nor segmentation")
                            })?),
                        };
                        let mut annotation = Annotation::new(class_id, shape);
                        annotation.id = a.id;
                        annotation.mask = mask;
                        if let Some(b) = bbox {
                            annotation.bbox = b;
                        }
                        if a.iscrowd > 1 || a.keypoints.len() % 3 != 0 {
                            return Err(invalid("invalid COCO crowd flag or keypoint triplets"));
                        }
                        annotation.is_crowd = a.iscrowd == 1;
                        annotation.metadata = a.metadata.clone();
                        for k in a.keypoints.as_chunks::<3>().0 {
                            if k[2].fract() != 0.0 || !(0.0..=2.0).contains(&k[2]) {
                                return Err(invalid("invalid COCO visibility"));
                            }
                            annotation.keypoints.push(Keypoint::new(
                                k[0],
                                k[1],
                                Visibility::try_from(k[2] as u8)?,
                            ));
                        }
                        sample.objects.push(annotation)?;
                    }
                }
                Ok(sample)
            })
            .collect();
        dataset.samples.extend(samples?);
    }
    dataset
        .metadata
        .insert("coco_sources".into(), source_metadata.into());
    dataset.validate()?;
    Ok(dataset)
}
/// Read the split layout generated by this adapter.
pub fn read_dir(root: impl AsRef<Path>) -> Result<Dataset> {
    let root = root.as_ref();
    let sources: Vec<_> = super::collect_files(&root.join("annotations"), "json")?
        .into_iter()
        .map(|p| Source {
            split: Split::from_name(
                p.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unassigned"),
            ),
            annotations: p,
            images: root.to_path_buf(),
        })
        .collect();
    let mut dataset = read(&sources)?;
    if root.join("metadata.json").is_file() {
        dataset.metadata = read_json(root.join("metadata.json"))?;
    }
    Ok(dataset)
}
pub fn write(
    dataset: &Dataset,
    root: impl AsRef<Path>,
    options: ExportOptions,
) -> Result<ExportReport> {
    dataset.validate()?;
    for c in &dataset.categories {
        super::check_metadata(&c.metadata, &["id", "name", "keypoints", "skeleton"])?;
    }
    for sample in &dataset.samples {
        for o in sample.objects.iter() {
            super::check_metadata(
                o.metadata(),
                &[
                    "id",
                    "image_id",
                    "category_id",
                    "bbox",
                    "segmentation",
                    "keypoints",
                    "iscrowd",
                    "area",
                    "num_keypoints",
                ],
            )?;
            if o.mask()
                .is_some_and(|m| m.width() as u64 * m.height() as u64 > u32::MAX as u64)
            {
                return Err(invalid("COCO RLE mask exceeds 32-bit pixel count"));
            }
        }
    }
    let root = root.as_ref();
    let mut report = ExportReport::default();
    let mut documents = BTreeMap::<String, Document>::new();
    let mut image_jobs = vec![];
    let mut annotation_id = 0_u64;
    let new_document = || Document {
        images: vec![],
        annotations: vec![],
        categories: dataset
            .categories
            .iter()
            .enumerate()
            .map(|(i, c)| CocoCategory {
                id: i as u64 + 1,
                name: c.name.clone(),
                keypoints: c.keypoints.clone(),
                skeleton: c.skeleton.iter().map(|[a, b]| [a + 1, b + 1]).collect(),
                metadata: c.metadata.clone(),
            })
            .collect(),
        metadata: Metadata::new(),
    };
    if dataset.samples.is_empty() {
        documents.insert("unassigned".into(), new_document());
    }
    for (i, s) in dataset.samples.iter().enumerate() {
        let split = super::split_name(s, options.splits)?;
        let doc = documents.entry(split.clone()).or_insert_with(&new_document);
        let name = format!("images/{split}/{}.png", s.uid);
        image_jobs.push((s, name.clone()));
        doc.images.push(CocoImage {
            id: i as u64 + 1,
            file_name: name,
            width: s.image.width(),
            height: s.image.height(),
            identity: Some(Identity::of(s, options.splits)),
            metadata: Metadata::new(),
        });
        for o in s.objects.iter() {
            let segmentation = if let Some(m) = o.mask() {
                Some(Segmentation::Rle {
                    size: [m.height(), m.width()],
                    counts: Counts::Compressed(encode_counts(&encode_mask(m))),
                })
            } else {
                match o.shape() {
                    Shape::Rect(_) => None,
                    Shape::Point(_) => {
                        report.loss(
                            options.loss,
                            format!("{}: standalone point has no COCO instance geometry", s.uid),
                        )?;
                        None
                    }
                    shape => {
                        if matches!(shape, Shape::Circle { .. }) {
                            report.loss(
                                options.loss,
                                format!("{}: circle approximated by 64 polygon vertices", s.uid),
                            )?;
                        }
                        Some(Segmentation::Polygons(
                            shape
                                .polygons(64)
                                .iter()
                                .map(|p| p.iter().flat_map(|p| [p.x, p.y]).collect())
                                .collect(),
                        ))
                    }
                }
            };
            let b = o.bbox();
            annotation_id += 1;
            doc.annotations.push(CocoAnnotation {
                id: annotation_id,
                image_id: i as u64 + 1,
                category_id: o.class_id() as u64 + 1,
                bbox: Some([b.x, b.y, b.width, b.height]),
                segmentation,
                keypoints: o
                    .keypoints()
                    .iter()
                    .flat_map(|k| [k.point.x, k.point.y, k.visibility as u8 as f32])
                    .collect(),
                iscrowd: u8::from(o.is_crowd()),
                area: o.area() as f64,
                num_keypoints: o
                    .keypoints()
                    .iter()
                    .filter(|k| k.visibility != Visibility::Absent)
                    .count() as u32,
                metadata: o.metadata().clone(),
            });
            report.annotations += 1;
        }
        report.samples += 1;
    }
    // All representability checks finish before creating output.
    super::prepare_output(root)?;
    std::fs::create_dir(root.join("annotations"))?;
    for (split, doc) in documents {
        std::fs::create_dir_all(root.join("images").join(&split))?;
        write_json(root.join("annotations").join(format!("{split}.json")), &doc)?;
    }
    image_jobs
        .par_iter()
        .try_for_each(|(s, name)| s.image.write_png(root.join(name)))?;
    write_json(root.join("metadata.json"), &dataset.metadata)?;
    Ok(report)
}

pub(crate) fn decode_mask(width: u32, height: u32, counts: &[u32]) -> Result<Mask> {
    let n = crate::ir::pixel_count(width, height)?;
    if counts.iter().map(|&v| v as u64).sum::<u64>() != n as u64 {
        return Err(invalid("COCO RLE counts do not sum to image area"));
    }
    let mut dense = vec![0; n];
    let mut cursor = 0;
    for (i, &run) in counts.iter().enumerate() {
        for column_major in cursor..cursor + run as usize {
            if i % 2 == 1 {
                let x = column_major / height as usize;
                let y = column_major % height as usize;
                dense[y * width as usize + x] = 1;
            }
        }
        cursor += run as usize;
    }
    Mask::from_dense(width, height, &dense)
}
fn encode_mask(mask: &Mask) -> Vec<u32> {
    let mut counts = vec![];
    let mut current = false;
    let mut run = 0;
    for x in 0..mask.width() {
        for y in 0..mask.height() {
            let value = mask.get(x, y);
            if value != current {
                counts.push(run);
                run = 0;
                current = value;
            }
            run += 1;
        }
    }
    counts.push(run);
    counts
}
/// COCO's signed 5-bit delta code. Counts after index 2 are deltas from i-2.
pub(crate) fn decode_counts(s: &str) -> Result<Vec<u32>> {
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut out = Vec::<u32>::new();
    while i < bytes.len() {
        let mut value = 0_i64;
        let mut shift = 0;
        loop {
            let c = *bytes
                .get(i)
                .ok_or_else(|| invalid("truncated COCO RLE count"))?;
            if !(48..=111).contains(&c) || shift >= 35 {
                return Err(invalid("invalid COCO compressed RLE"));
            }
            i += 1;
            let c = c - 48;
            value |= ((c & 31) as i64) << shift;
            shift += 5;
            if c & 32 == 0 {
                if c & 16 != 0 {
                    value |= (-1_i64) << shift;
                }
                break;
            }
        }
        if out.len() > 2 {
            value += out[out.len() - 2] as i64;
        }
        if !(0..=u32::MAX as i64).contains(&value) {
            return Err(invalid("COCO RLE count overflow"));
        }
        out.push(value as u32);
    }
    Ok(out)
}
fn encode_counts(counts: &[u32]) -> String {
    let mut bytes = vec![];
    for (i, &n) in counts.iter().enumerate() {
        let mut x = n as i64;
        if i > 2 {
            x -= counts[i - 2] as i64;
        }
        loop {
            let mut c = (x & 31) as u8;
            x >>= 5;
            let more = if c & 16 != 0 { x != -1 } else { x != 0 };
            if more {
                c |= 32;
            }
            bytes.push(c + 48);
            if !more {
                break;
            }
        }
    }
    String::from_utf8(bytes).expect("ASCII RLE")
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Deserialize)]
    struct Golden {
        width: u32,
        height: u32,
        dense: Vec<u8>,
        counts: String,
        area: u64,
    }
    #[test]
    fn pycocotools_golden_masks() {
        // Generated independently with pycocotools 2.0.11, NumPy seed 42.
        let cases: Vec<Golden> =
            serde_json::from_str(include_str!("../../tests/fixtures/coco_rle.json")).unwrap();
        for case in cases {
            let mask = decode_mask(
                case.width,
                case.height,
                &decode_counts(&case.counts).unwrap(),
            )
            .unwrap();
            assert_eq!(mask.to_dense(), case.dense);
            assert_eq!(mask.area(), case.area);
            assert_eq!(encode_counts(&encode_mask(&mask)), case.counts);
        }
    }
    #[test]
    fn signed_rle_deltas() {
        for c in [
            vec![0, 1, 9, 2, 7, 30, 1, 0, 12345],
            vec![u32::MAX, 0, 1, 0],
        ] {
            assert_eq!(decode_counts(&encode_counts(&c)).unwrap(), c);
        }
        assert!(decode_counts("P").is_err());
        assert!(decode_counts("!").is_err());
    }
}
