use crate::{Point, Rect, Result, Shape, invalid};
use image::ImageEncoder;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    ops::Range,
    path::Path,
};

pub type Metadata = BTreeMap<String, serde_json::Value>;

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Split {
    Train,
    Val,
    Test,
    #[default]
    Unassigned,
    Named(String),
}
impl Split {
    pub fn from_name(name: &str) -> Self {
        match name {
            "train" => Self::Train,
            "val" | "valid" | "validation" => Self::Val,
            "test" => Self::Test,
            "unassigned" => Self::Unassigned,
            s => Self::Named(s.to_owned()),
        }
    }
    pub fn as_str(&self) -> &str {
        match self {
            Self::Train => "train",
            Self::Val => "val",
            Self::Test => "test",
            Self::Unassigned => "unassigned",
            Self::Named(s) => s,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Category {
    /// Dense, zero-based class index in the IR. Source IDs belong in metadata.
    pub name: String,
    pub keypoints: Vec<String>,
    /// Zero-based keypoint index pairs (COCO adapters convert from/to one-based).
    pub skeleton: Vec<[u32; 2]>,
    pub metadata: Metadata,
}
impl Category {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            keypoints: vec![],
            skeleton: vec![],
            metadata: Metadata::new(),
        }
    }
}

/// Decoded, tightly packed row-major RGB8. No paths or external image dependencies.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Raster {
    width: u32,
    height: u32,
    #[serde(with = "serde_bytes")]
    pixels: Vec<u8>,
}
impl Raster {
    pub fn new(width: u32, height: u32, pixels: Vec<u8>) -> Result<Self> {
        let r = Self {
            width,
            height,
            pixels,
        };
        r.validate()?;
        Ok(r)
    }
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let rgb = image::ImageReader::open(path)?
            .with_guessed_format()?
            .decode()?
            .into_rgb8();
        Self::new(rgb.width(), rgb.height(), rgb.into_raw())
    }
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let rgb = image::load_from_memory(bytes)?.into_rgb8();
        Self::new(rgb.width(), rgb.height(), rgb.into_raw())
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }
    pub fn pixels_mut(&mut self) -> &mut [u8] {
        &mut self.pixels
    }
    /// Transfer the contiguous RGB allocation to a foreign-language array owner.
    pub fn into_pixels(self) -> Vec<u8> {
        self.pixels
    }
    pub fn bounds(&self) -> Rect {
        Rect::new(0.0, 0.0, self.width as f32, self.height as f32)
    }
    pub fn validate(&self) -> Result<()> {
        if pixel_count(self.width, self.height)?.checked_mul(3) != Some(self.pixels.len()) {
            return Err(invalid("RGB buffer length does not match image dimensions"));
        }
        Ok(())
    }
    pub fn write_png(&self, path: impl AsRef<Path>) -> Result<()> {
        use std::io::Write;
        let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
        image::codecs::png::PngEncoder::new(&mut out).write_image(
            &self.pixels,
            self.width,
            self.height,
            image::ExtendedColorType::Rgb8,
        )?;
        out.flush()?;
        Ok(())
    }
}
pub(crate) fn pixel_count(w: u32, h: u32) -> Result<usize> {
    if w == 0 || h == 0 {
        return Err(invalid("image dimensions must be positive"));
    }
    (w as usize)
        .checked_mul(h as usize)
        .filter(|n| *n <= isize::MAX as usize / 3)
        .ok_or_else(|| invalid("image dimensions overflow"))
}

/// One bit per pixel, row-major, least significant bit first. Holes and islands are exact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mask {
    width: u32,
    height: u32,
    #[serde(with = "serde_bytes")]
    bits: Vec<u8>,
}
impl Mask {
    pub fn from_bits(width: u32, height: u32, bits: Vec<u8>) -> Result<Self> {
        let mask = Self {
            width,
            height,
            bits,
        };
        mask.validate()?;
        Ok(mask)
    }
    pub fn bits(&self) -> &[u8] {
        &self.bits
    }
    pub fn from_dense(width: u32, height: u32, data: &[u8]) -> Result<Self> {
        let n = pixel_count(width, height)?;
        if data.len() != n {
            return Err(invalid("mask length mismatch"));
        }
        let mut bits = vec![0; n.div_ceil(8)];
        for (i, &v) in data.iter().enumerate() {
            if v != 0 {
                bits[i / 8] |= 1 << (i % 8);
            }
        }
        Ok(Self {
            width,
            height,
            bits,
        })
    }
    pub fn from_fn(width: u32, height: u32, mut f: impl FnMut(u32, u32) -> bool) -> Result<Self> {
        let n = pixel_count(width, height)?;
        let mut bits = vec![0; n.div_ceil(8)];
        for i in 0..n {
            if f((i % width as usize) as u32, (i / width as usize) as u32) {
                bits[i / 8] |= 1 << (i % 8);
            }
        }
        Ok(Self {
            width,
            height,
            bits,
        })
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn get(&self, x: u32, y: u32) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        let i = y as usize * self.width as usize + x as usize;
        self.bits[i / 8] & (1 << (i % 8)) != 0
    }
    pub fn area(&self) -> u64 {
        self.bits.iter().map(|b| b.count_ones() as u64).sum()
    }
    pub fn to_dense(&self) -> Vec<u8> {
        (0..self.width as usize * self.height as usize)
            .map(|i| u8::from(self.bits[i / 8] & (1 << (i % 8)) != 0))
            .collect()
    }
    /// Exact integer `(x, y, width, height)` extent, or `None` for an empty mask.
    pub fn pixel_bounds(&self) -> Option<(u32, u32, u32, u32)> {
        let (mut x0, mut y0, mut x1, mut y1) = (self.width, self.height, 0, 0);
        for y in 0..self.height {
            for x in 0..self.width {
                if self.get(x, y) {
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x + 1);
                    y1 = y1.max(y + 1);
                }
            }
        }
        (x1 > x0 && y1 > y0).then(|| (x0, y0, x1 - x0, y1 - y0))
    }
    pub fn bounds(&self) -> Option<Rect> {
        self.pixel_bounds()
            .map(|(x, y, w, h)| Rect::new(x as f32, y as f32, w as f32, h as f32))
    }
    pub fn validate(&self) -> Result<()> {
        let n = pixel_count(self.width, self.height)?;
        if self.bits.len() != n.div_ceil(8)
            || (n % 8 != 0 && self.bits.last().is_some_and(|b| *b >> (n % 8) != 0))
        {
            return Err(invalid("invalid packed mask"));
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Visibility {
    Absent = 0,
    Occluded = 1,
    Visible = 2,
}
impl TryFrom<u8> for Visibility {
    type Error = crate::Error;
    fn try_from(v: u8) -> Result<Self> {
        match v {
            0 => Ok(Self::Absent),
            1 => Ok(Self::Occluded),
            2 => Ok(Self::Visible),
            _ => Err(invalid("keypoint visibility must be 0, 1 or 2")),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct Keypoint {
    pub point: Point,
    pub visibility: Visibility,
}
impl Keypoint {
    pub const fn new(x: f32, y: f32, visibility: Visibility) -> Self {
        Self {
            point: Point::new(x, y),
            visibility,
        }
    }
}

/// Convenient owned value at API boundaries; stored column-wise in ObjectTable.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Annotation {
    pub id: u64,
    pub class_id: u32,
    pub bbox: Rect,
    pub shape: Shape,
    pub mask: Option<Mask>,
    pub keypoints: Vec<Keypoint>,
    pub is_crowd: bool,
    pub metadata: Metadata,
}
impl Annotation {
    pub fn new(class_id: u32, shape: Shape) -> Self {
        Self {
            id: 0,
            class_id,
            bbox: shape.bounds(),
            shape,
            mask: None,
            keypoints: vec![],
            is_crowd: false,
            metadata: Metadata::new(),
        }
    }
    pub fn validate(&self) -> Result<()> {
        self.bbox.validate()?;
        self.shape.validate()?;
        if let Some(m) = &self.mask {
            m.validate()?;
        }
        if self.keypoints.iter().any(|p| !p.point.is_finite()) {
            return Err(invalid("non-finite keypoint"));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum PackedShape {
    Rect(Rect),
    Circle { center: Point, radius: f32 },
    Polygons(Range<usize>),
    Point(Point),
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ObjectTable {
    ids: Vec<u64>,
    class_ids: Vec<u32>,
    boxes: Vec<Rect>,
    shapes: Vec<PackedShape>,
    vertices: Vec<Point>,
    polygons: Vec<Range<usize>>,
    keypoints: Vec<Keypoint>,
    keypoint_ranges: Vec<Range<usize>>,
    masks: Vec<Option<Mask>>,
    crowds: Vec<bool>,
    metadata: Vec<Metadata>,
}
impl ObjectTable {
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    pub fn class_ids(&self) -> &[u32] {
        &self.class_ids
    }
    pub fn boxes(&self) -> &[Rect] {
        &self.boxes
    }
    pub fn vertices(&self) -> &[Point] {
        &self.vertices
    }
    pub fn keypoints(&self) -> &[Keypoint] {
        &self.keypoints
    }
    pub fn get(&self, index: usize) -> Option<ObjectRef<'_>> {
        (index < self.len()).then_some(ObjectRef { table: self, index })
    }
    pub fn iter(&self) -> impl ExactSizeIterator<Item = ObjectRef<'_>> {
        (0..self.len()).map(|index| ObjectRef { table: self, index })
    }
    pub fn push(&mut self, a: Annotation) -> Result<()> {
        a.validate()?;
        let shape = match a.shape {
            Shape::Rect(r) => PackedShape::Rect(r),
            Shape::Circle { center, radius } => PackedShape::Circle { center, radius },
            Shape::Point(p) => PackedShape::Point(p),
            Shape::Polygons(parts) => {
                let start = self.polygons.len();
                for p in parts {
                    let s = self.vertices.len();
                    self.vertices.extend(p);
                    self.polygons.push(s..self.vertices.len());
                }
                PackedShape::Polygons(start..self.polygons.len())
            }
        };
        self.ids.push(a.id);
        self.class_ids.push(a.class_id);
        self.boxes.push(a.bbox);
        self.shapes.push(shape);
        let start = self.keypoints.len();
        self.keypoints.extend(a.keypoints);
        self.keypoint_ranges.push(start..self.keypoints.len());
        self.masks.push(a.mask);
        self.crowds.push(a.is_crowd);
        self.metadata.push(a.metadata);
        Ok(())
    }
    /// Stable selection, compacting all auxiliary pools and preserving instance alignment.
    pub fn retain(&mut self, mut keep: impl FnMut(ObjectRef<'_>) -> bool) -> Result<()> {
        let selected: Vec<bool> = self.iter().map(&mut keep).collect();
        if selected.iter().all(|&v| v) {
            return Ok(());
        }
        if selected.iter().all(|&v| !v) {
            *self = Self::default();
            return Ok(());
        }
        let mut old = std::mem::take(self);
        let masks = std::mem::take(&mut old.masks);
        let metadata = std::mem::take(&mut old.metadata);
        let mut out = Self::default();
        for (index, (mask, metadata)) in masks.into_iter().zip(metadata).enumerate() {
            if selected[index] {
                let object = ObjectRef { table: &old, index };
                out.push(Annotation {
                    id: object.id(),
                    class_id: object.class_id(),
                    bbox: object.bbox(),
                    shape: object.shape(),
                    mask,
                    keypoints: object.keypoints().to_vec(),
                    is_crowd: object.is_crowd(),
                    metadata,
                })?;
            }
        }
        *self = out;
        Ok(())
    }
    pub fn validate(&self) -> Result<()> {
        let n = self.len();
        if [
            self.class_ids.len(),
            self.boxes.len(),
            self.shapes.len(),
            self.keypoint_ranges.len(),
            self.masks.len(),
            self.crowds.len(),
            self.metadata.len(),
        ]
        .iter()
        .any(|l| *l != n)
        {
            return Err(invalid("annotation columns have different lengths"));
        }
        let valid = |r: &Range<usize>, n: usize| r.start <= r.end && r.end <= n;
        if self.polygons.iter().any(|r| !valid(r, self.vertices.len()))
            || self
                .keypoint_ranges
                .iter()
                .any(|r| !valid(r, self.keypoints.len()))
            || self
                .shapes
                .iter()
                .any(|s| matches!(s,PackedShape::Polygons(r) if !valid(r,self.polygons.len())))
        {
            return Err(invalid("annotation pool range out of bounds"));
        }
        for o in self.iter() {
            o.bbox().validate()?;
            if o.keypoints().iter().any(|k| !k.point.is_finite()) {
                return Err(invalid("non-finite keypoint"));
            }
            if let Some(mask) = o.mask() {
                mask.validate()?;
            }
            match &self.shapes[o.index] {
                PackedShape::Polygons(r) => {
                    if r.is_empty() {
                        return Err(invalid("empty polygon collection"));
                    }
                    for p in o.polygons() {
                        if p.len() < 3
                            || p.iter().any(|p| !p.is_finite())
                            || crate::geometry::polygon_area(p) <= 0.0
                        {
                            return Err(invalid("invalid polygon"));
                        }
                        Rect::from_points(p).validate()?;
                    }
                    if !o.area().is_finite() {
                        return Err(invalid("polygon area overflow"));
                    }
                }
                _ => o.shape().validate()?,
            }
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug)]
pub struct ObjectRef<'a> {
    table: &'a ObjectTable,
    index: usize,
}
impl<'a> ObjectRef<'a> {
    pub fn id(self) -> u64 {
        self.table.ids[self.index]
    }
    pub fn class_id(self) -> u32 {
        self.table.class_ids[self.index]
    }
    pub fn bbox(self) -> Rect {
        self.table.boxes[self.index]
    }
    pub fn keypoints(self) -> &'a [Keypoint] {
        &self.table.keypoints[self.table.keypoint_ranges[self.index].clone()]
    }
    pub fn mask(self) -> Option<&'a Mask> {
        self.table.masks[self.index].as_ref()
    }
    pub fn is_crowd(self) -> bool {
        self.table.crowds[self.index]
    }
    pub fn metadata(self) -> &'a Metadata {
        &self.table.metadata[self.index]
    }
    pub fn shape(self) -> Shape {
        match &self.table.shapes[self.index] {
            PackedShape::Rect(r) => Shape::Rect(*r),
            PackedShape::Circle { center, radius } => Shape::Circle {
                center: *center,
                radius: *radius,
            },
            PackedShape::Point(p) => Shape::Point(*p),
            PackedShape::Polygons(r) => Shape::Polygons(
                self.table.polygons[r.clone()]
                    .iter()
                    .map(|r| self.table.vertices[r.clone()].to_vec())
                    .collect(),
            ),
        }
    }
    /// Borrow polygon vertex slices without allocation. Empty for other primitives.
    pub fn polygons(self) -> impl Iterator<Item = &'a [Point]> {
        let r = match &self.table.shapes[self.index] {
            PackedShape::Polygons(r) => r.clone(),
            _ => 0..0,
        };
        self.table.polygons[r]
            .iter()
            .map(|r| &self.table.vertices[r.clone()])
    }
    pub fn area(self) -> f32 {
        if let Some(m) = self.mask() {
            return m.area() as f32;
        }
        match &self.table.shapes[self.index] {
            PackedShape::Rect(r) => r.area(),
            PackedShape::Circle { radius, .. } => std::f32::consts::PI * radius * radius,
            PackedShape::Point(_) => 0.0,
            PackedShape::Polygons(_) => self.polygons().map(crate::geometry::polygon_area).sum(),
        }
    }
    pub fn to_owned(self) -> Annotation {
        Annotation {
            id: self.id(),
            class_id: self.class_id(),
            bbox: self.bbox(),
            shape: self.shape(),
            mask: self.mask().cloned(),
            keypoints: self.keypoints().to_vec(),
            is_crowd: self.is_crowd(),
            metadata: self.metadata().clone(),
        }
    }
}
/// Transform history is separate from user metadata, so metadata keys are never overwritten.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub parent_uid: uuid::Uuid,
    pub operation: String,
    pub parameters: Metadata,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sample {
    pub uid: uuid::Uuid,
    pub id: String,
    pub image: Raster,
    pub objects: ObjectTable,
    pub split: Split,
    pub metadata: Metadata,
    #[serde(default)]
    pub provenance: Vec<Provenance>,
}
impl Sample {
    pub fn new(id: impl Into<String>, image: Raster) -> Self {
        Self {
            uid: uuid::Uuid::new_v4(),
            id: id.into(),
            image,
            objects: ObjectTable::default(),
            split: Split::Unassigned,
            metadata: Metadata::new(),
            provenance: vec![],
        }
    }
    pub fn validate(&self, categories: &[Category]) -> Result<()> {
        if self.id.is_empty() {
            return Err(invalid("sample ID must not be empty"));
        }
        self.image.validate()?;
        self.objects.validate()?;
        for o in self.objects.iter() {
            let category = categories.get(o.class_id() as usize).ok_or_else(|| {
                invalid(format!(
                    "sample {} has unknown class {}",
                    self.id,
                    o.class_id()
                ))
            })?;
            if !o.keypoints().is_empty() && o.keypoints().len() != category.keypoints.len() {
                return Err(invalid(format!(
                    "sample {} has wrong keypoint count for {}",
                    self.id, category.name
                )));
            }
            if let Some(m) = o.mask()
                && (m.width(), m.height()) != (self.image.width(), self.image.height())
            {
                return Err(invalid("mask dimensions do not match image"));
            }
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Dataset {
    pub categories: Vec<Category>,
    pub samples: Vec<Sample>,
    pub metadata: Metadata,
}
impl Dataset {
    pub fn validate(&self) -> Result<()> {
        let mut names = HashSet::new();
        for c in &self.categories {
            if c.name.is_empty() || !names.insert(&c.name) {
                return Err(invalid("category names must be nonempty and unique"));
            }
            let points: HashSet<_> = c.keypoints.iter().collect();
            if points.len() != c.keypoints.len()
                || c.keypoints.iter().any(String::is_empty)
                || c.skeleton
                    .iter()
                    .flatten()
                    .any(|i| *i as usize >= c.keypoints.len())
            {
                return Err(invalid("invalid keypoint schema"));
            }
        }
        let mut uids = HashSet::new();
        let mut ids = HashSet::new();
        for s in &self.samples {
            if s.uid.is_nil() || !uids.insert(s.uid) {
                return Err(invalid("nil or duplicate sample UUID"));
            }
            if !ids.insert(&s.id) {
                return Err(invalid(format!("duplicate sample ID {}", s.id)));
            }
            s.validate(&self.categories)?;
        }
        Ok(())
    }
    pub fn ignore_splits(&mut self) {
        for s in &mut self.samples {
            s.split = Split::Unassigned;
        }
    }
    pub fn retain_split(&mut self, split: &Split) {
        self.samples.retain(|s| &s.split == split);
    }
}

/// Borrowed index; Rust prevents mutation of the dataset while this index exists.
/// Construction is O(samples), lookup is expected O(1), no image data is copied.
pub struct DatasetIndex<'a> {
    dataset: &'a Dataset,
    positions: HashMap<uuid::Uuid, usize>,
}
impl Dataset {
    pub fn index(&self) -> Result<DatasetIndex<'_>> {
        let mut positions = HashMap::with_capacity(self.samples.len());
        for (i, s) in self.samples.iter().enumerate() {
            if s.uid.is_nil() || positions.insert(s.uid, i).is_some() {
                return Err(invalid("nil or duplicate sample UUID"));
            }
        }
        Ok(DatasetIndex {
            dataset: self,
            positions,
        })
    }
}
impl<'a> DatasetIndex<'a> {
    pub fn get(&self, uid: uuid::Uuid) -> Option<&'a Sample> {
        self.positions.get(&uid).map(|&i| &self.dataset.samples[i])
    }
    pub fn get_u128(&self, uid: u128) -> Option<&'a Sample> {
        self.get(uuid::Uuid::from_u128(uid))
    }
}
