//! Owned, parallel sample passes. Coordinates and pixels are transformed together.
use crate::{
    Annotation, Dataset, Keypoint, Mask, ObjectRef, ObjectTable, Point, Raster, Rect, Result,
    Sample, Shape, Visibility, invalid,
};
use rayon::prelude::*;

/// A pass can keep, drop, or expand a sample. Implement this trait for custom passes.
pub trait Pass: Send + Sync {
    fn apply(&self, sample: Sample) -> Result<Vec<Sample>>;
}
#[derive(Default)]
pub struct Pipeline {
    passes: Vec<Box<dyn Pass>>,
}
impl Pipeline {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn then(mut self, pass: impl Pass + 'static) -> Self {
        self.passes.push(Box::new(pass));
        self
    }
    /// Order is stable regardless of Rayon thread count; derivations receive new UUIDs.
    /// Install in a caller-owned Rayon pool to control the thread count.
    pub fn run(&self, mut dataset: Dataset) -> Result<Dataset> {
        dataset.validate()?;
        for pass in &self.passes {
            let batches: Result<Vec<_>> = dataset
                .samples
                .into_par_iter()
                .map(|s| pass.apply(s))
                .collect();
            dataset.samples = batches?.into_iter().flatten().collect();
        }
        dataset.validate()?;
        Ok(dataset)
    }
    /// Process one sample, suitable for a bounded-memory packed-file pipeline.
    pub fn run_sample(&self, sample: Sample) -> Result<Vec<Sample>> {
        let mut samples = vec![sample];
        for pass in &self.passes {
            let mut next = Vec::new();
            for s in samples {
                next.extend(pass.apply(s)?);
            }
            samples = next;
        }
        Ok(samples)
    }
}
/// Stable object filtering; by default empty/background images are retained.
pub struct FilterObjects<F> {
    pub predicate: F,
    pub drop_empty: bool,
}
impl<F> FilterObjects<F> {
    pub fn new(predicate: F) -> Self {
        Self {
            predicate,
            drop_empty: false,
        }
    }
}
impl<F: Fn(ObjectRef<'_>) -> bool + Send + Sync> Pass for FilterObjects<F> {
    fn apply(&self, mut sample: Sample) -> Result<Vec<Sample>> {
        sample.objects.retain(&self.predicate)?;
        if self.drop_empty && sample.objects.is_empty() {
            Ok(vec![])
        } else {
            Ok(vec![sample])
        }
    }
}
/// Bounding-box area and class filter without allocating polygon geometry.
#[derive(Clone, Debug)]
pub struct Filter {
    pub classes: Option<Vec<u32>>,
    pub min_area: f32,
    pub max_area: f32,
    pub drop_empty: bool,
}
impl Default for Filter {
    fn default() -> Self {
        Self {
            classes: None,
            min_area: 0.0,
            max_area: f32::INFINITY,
            drop_empty: false,
        }
    }
}
impl Pass for Filter {
    fn apply(&self, mut s: Sample) -> Result<Vec<Sample>> {
        if !self.min_area.is_finite()
            || self.min_area < 0.0
            || self.max_area.is_nan()
            || self.max_area < self.min_area
        {
            return Err(invalid("invalid area filter"));
        }
        s.objects.retain(|o| {
            self.classes
                .as_ref()
                .is_none_or(|c| c.contains(&o.class_id()))
                && o.bbox().area() >= self.min_area
                && o.bbox().area() <= self.max_area
        })?;
        if self.drop_empty && s.objects.is_empty() {
            Ok(vec![])
        } else {
            Ok(vec![s])
        }
    }
}
/// Affine map of pixel edges: `(x',y') = (a*x+b*y+tx, c*x+d*y+ty)`.
#[derive(Clone, Copy, Debug)]
pub struct Affine {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub tx: f32,
    pub ty: f32,
}
impl Affine {
    pub const IDENTITY: Self = Self {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 1.0,
        tx: 0.0,
        ty: 0.0,
    };
    pub fn map(self, p: Point) -> Point {
        Point::new(
            self.a * p.x + self.b * p.y + self.tx,
            self.c * p.x + self.d * p.y + self.ty,
        )
    }
    pub fn inverse(self) -> Result<Self> {
        let det = self.a * self.d - self.b * self.c;
        if ![self.a, self.b, self.c, self.d, self.tx, self.ty, det]
            .iter()
            .all(|x| x.is_finite())
            || det.abs() < 1e-12
        {
            return Err(invalid("affine transform must be finite and invertible"));
        }
        let (a, b, c, d) = (self.d / det, -self.b / det, -self.c / det, self.a / det);
        Ok(Self {
            a,
            b,
            c,
            d,
            tx: -a * self.tx - b * self.ty,
            ty: -c * self.tx - d * self.ty,
        })
    }
    pub fn shape(self, s: &Shape) -> Shape {
        let scale = self
            .a
            .abs()
            .max(self.b.abs())
            .max(self.c.abs())
            .max(self.d.abs());
        let tolerance = scale * 1e-7;
        let (a, b, c, d) = (self.a as f64, self.b as f64, self.c as f64, self.d as f64);
        let (sx2, sy2, dot) = (a * a + c * c, b * b + d * d, a * b + c * d);
        let similarity =
            (sx2 - sy2).abs() <= 1e-6 * sx2.max(sy2) && dot.abs() <= 1e-6 * (sx2 * sy2).sqrt();
        match s {
            Shape::Point(p) => Shape::Point(self.map(*p)),
            Shape::Rect(r)
                if (self.b.abs() <= tolerance && self.c.abs() <= tolerance)
                    || (self.a.abs() <= tolerance && self.d.abs() <= tolerance) =>
            {
                Shape::Rect(Rect::from_points(
                    &r.corners()
                        .into_iter()
                        .map(|p| self.map(p))
                        .collect::<Vec<_>>(),
                ))
            }
            Shape::Circle { center, radius } if similarity => Shape::Circle {
                center: self.map(*center),
                radius: radius * self.a.hypot(self.c),
            },
            _ => Shape::Polygons(
                s.polygons(64)
                    .into_iter()
                    .map(|p| p.into_iter().map(|p| self.map(p)).collect())
                    .collect(),
            ),
        }
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub enum Interpolation {
    Nearest,
    #[default]
    Bilinear,
}
#[derive(Clone, Debug)]
pub struct Warp {
    pub affine: Affine,
    pub width: u32,
    pub height: u32,
    pub fill: [u8; 3],
    pub interpolation: Interpolation,
}
impl Pass for Warp {
    fn apply(&self, s: Sample) -> Result<Vec<Sample>> {
        Ok(vec![self.transform(&s, "warp")?])
    }
}
impl Warp {
    pub fn transform(&self, s: &Sample, operation: &str) -> Result<Sample> {
        let inverse = self.affine.inverse()?;
        let n = crate::ir::pixel_count(self.width, self.height)?;
        let mut pixels = vec![0; n * 3];
        pixels
            .par_chunks_mut(self.width as usize * 3)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, out) in row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                    let p = inverse.map(Point::new(x as f32 + 0.5, y as f32 + 0.5));
                    let rgb = sample_pixel(&s.image, p, self.interpolation, self.fill);
                    out.copy_from_slice(&rgb);
                }
            });
        finish(
            s,
            Raster::new(self.width, self.height, pixels)?,
            self.affine,
            operation,
            0.0,
            None,
        )
    }
}
fn sample_pixel(image: &Raster, p: Point, interpolation: Interpolation, fill: [u8; 3]) -> [u8; 3] {
    let at = |x: i64, y: i64| {
        if x < 0 || y < 0 || x >= image.width() as i64 || y >= image.height() as i64 {
            fill
        } else {
            let i = (y as usize * image.width() as usize + x as usize) * 3;
            [
                image.pixels()[i],
                image.pixels()[i + 1],
                image.pixels()[i + 2],
            ]
        }
    };
    match interpolation {
        Interpolation::Nearest => at(p.x.floor() as i64, p.y.floor() as i64),
        Interpolation::Bilinear => {
            let x = p.x - 0.5;
            let y = p.y - 0.5;
            let ix = x.floor() as i64;
            let iy = y.floor() as i64;
            let fx = x - x.floor();
            let fy = y - y.floor();
            let a = at(ix, iy);
            let b = at(ix + 1, iy);
            let c = at(ix, iy + 1);
            let d = at(ix + 1, iy + 1);
            std::array::from_fn(|i| {
                ((a[i] as f32 * (1.0 - fx) + b[i] as f32 * fx) * (1.0 - fy)
                    + (c[i] as f32 * (1.0 - fx) + d[i] as f32 * fx) * fy)
                    .round()
                    .clamp(0.0, 255.0) as u8
            })
        }
    }
}
fn finish(
    s: &Sample,
    image: Raster,
    affine: Affine,
    operation: &str,
    min_visibility: f32,
    selected: Option<usize>,
) -> Result<Sample> {
    let inverse = affine.inverse()?;
    let bounds = image.bounds();
    let mut objects = ObjectTable::default();
    for (index, o) in s.objects.iter().enumerate() {
        if selected.is_some_and(|i| i != index) {
            continue;
        }
        let before = affine.shape(&o.shape());
        let Some(shape) = before.clip(bounds) else {
            continue;
        };
        if before.area() > 0.0 && shape.area() / before.area() < min_visibility {
            continue;
        }
        let mut a = Annotation::new(o.class_id(), shape);
        a.id = o.id();
        a.metadata = o.metadata().clone();
        a.is_crowd = o.is_crowd();
        // COCO boxes may differ from segmentation bounds: preserve their independent meaning.
        a.bbox = affine
            .shape(&Shape::Rect(o.bbox()))
            .bounds()
            .intersection(bounds)
            .unwrap_or(a.shape.bounds());
        if let Some(mask) = o.mask() {
            let m = Mask::from_fn(image.width(), image.height(), |x, y| {
                let p = inverse.map(Point::new(x as f32 + 0.5, y as f32 + 0.5));
                p.x >= 0.0 && p.y >= 0.0 && mask.get(p.x.floor() as u32, p.y.floor() as u32)
            })?;
            let Some(b) = m.bounds() else {
                continue;
            };
            a.bbox = b;
            a.mask = Some(m);
        }
        a.keypoints = o
            .keypoints()
            .iter()
            .map(|k| {
                let p = affine.map(k.point);
                if k.visibility == Visibility::Absent
                    || (p.x < 0.0 || p.y < 0.0 || p.x > bounds.right() || p.y > bounds.bottom())
                {
                    Keypoint::new(0.0, 0.0, Visibility::Absent)
                } else {
                    Keypoint {
                        point: p,
                        visibility: k.visibility,
                    }
                }
            })
            .collect();
        objects.push(a)?;
    }
    let parameters = crate::Metadata::from([
        (
            "affine".into(),
            serde_json::json!([affine.a, affine.b, affine.c, affine.d, affine.tx, affine.ty]),
        ),
        (
            "source_size".into(),
            serde_json::json!([s.image.width(), s.image.height()]),
        ),
        (
            "output_size".into(),
            serde_json::json!([image.width(), image.height()]),
        ),
    ]);
    Ok(derived(s, image, objects, operation, parameters))
}
/// Integer crop within the source image. Clips geometry; visibility uses shape area.
#[derive(Clone, Copy, Debug)]
pub struct Crop {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub min_visibility: f32,
}
impl Crop {
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            min_visibility: 0.0,
        }
    }
    pub fn transform(&self, s: &Sample) -> Result<Sample> {
        self.transform_selected(s, None)
    }
    fn transform_selected(&self, s: &Sample, selected: Option<usize>) -> Result<Sample> {
        if self.width == 0
            || self.height == 0
            || self
                .x
                .checked_add(self.width)
                .is_none_or(|x| x > s.image.width())
            || self
                .y
                .checked_add(self.height)
                .is_none_or(|y| y > s.image.height())
            || !(0.0..=1.0).contains(&self.min_visibility)
        {
            return Err(invalid(
                "crop is outside the image or visibility is invalid",
            ));
        }
        let mut pixels = vec![0; crate::ir::pixel_count(self.width, self.height)? * 3];
        for (dy, row) in pixels.chunks_exact_mut(self.width as usize * 3).enumerate() {
            let start = ((self.y as usize + dy) * s.image.width() as usize + self.x as usize) * 3;
            row.copy_from_slice(&s.image.pixels()[start..start + row.len()]);
        }
        finish(
            s,
            Raster::new(self.width, self.height, pixels)?,
            Affine {
                tx: -(self.x as f32),
                ty: -(self.y as f32),
                ..Affine::IDENTITY
            },
            "crop",
            self.min_visibility,
            selected,
        )
    }
}
impl Pass for Crop {
    fn apply(&self, s: Sample) -> Result<Vec<Sample>> {
        Ok(vec![self.transform(&s)?])
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Resize {
    pub width: u32,
    pub height: u32,
}
impl Pass for Resize {
    fn apply(&self, s: Sample) -> Result<Vec<Sample>> {
        use fast_image_resize::{
            FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer,
            images::{Image, ImageRef},
        };
        crate::ir::pixel_count(self.width, self.height)?;
        let src = ImageRef::new(
            s.image.width(),
            s.image.height(),
            s.image.pixels(),
            PixelType::U8x3,
        )
        .map_err(|e| invalid(e.to_string()))?;
        let mut dst = Image::new(self.width, self.height, PixelType::U8x3);
        Resizer::new()
            .resize(
                &src,
                &mut dst,
                &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)),
            )
            .map_err(|e| invalid(e.to_string()))?;
        let affine = Affine {
            a: self.width as f32 / s.image.width() as f32,
            d: self.height as f32 / s.image.height() as f32,
            ..Affine::IDENTITY
        };
        Ok(vec![finish(
            &s,
            Raster::new(self.width, self.height, dst.into_vec())?,
            affine,
            "resize",
            0.0,
            None,
        )?])
    }
}
/// Clockwise rotation in image coordinates. Canvas size is unchanged; outside is clipped.
#[derive(Clone, Copy, Debug)]
pub struct Rotate {
    pub degrees: f32,
    pub fill: [u8; 3],
}
impl Pass for Rotate {
    fn apply(&self, s: Sample) -> Result<Vec<Sample>> {
        if !self.degrees.is_finite() {
            return Err(invalid("rotation must be finite"));
        }
        let (sin, cos) = self.degrees.to_radians().sin_cos();
        let x = s.image.width() as f32 / 2.0;
        let y = s.image.height() as f32 / 2.0;
        let affine = Affine {
            a: cos,
            b: -sin,
            c: sin,
            d: cos,
            tx: x - cos * x + sin * y,
            ty: y - sin * x - cos * y,
        };
        Ok(vec![
            Warp {
                affine,
                width: s.image.width(),
                height: s.image.height(),
                fill: self.fill,
                interpolation: Interpolation::Bilinear,
            }
            .transform(&s, "rotate")?,
        ])
    }
}
/// Centered zoom on the original canvas. Factors below one expose the fill color.
#[derive(Clone, Copy, Debug)]
pub struct Zoom {
    pub factor: f32,
    pub fill: [u8; 3],
}
impl Pass for Zoom {
    fn apply(&self, s: Sample) -> Result<Vec<Sample>> {
        if !self.factor.is_finite() || self.factor <= 0.0 {
            return Err(invalid("zoom factor must be positive"));
        }
        let affine = Affine {
            a: self.factor,
            d: self.factor,
            tx: s.image.width() as f32 * (1.0 - self.factor) / 2.0,
            ty: s.image.height() as f32 * (1.0 - self.factor) / 2.0,
            ..Affine::IDENTITY
        };
        Ok(vec![
            Warp {
                affine,
                width: s.image.width(),
                height: s.image.height(),
                fill: self.fill,
                interpolation: Interpolation::Bilinear,
            }
            .transform(&s, "zoom")?,
        ])
    }
}
/// Horizontal flip. Optional permutation maps each output pose slot to its input slot.
/// Supply a complete involution (left/right swap) to preserve anatomical semantics.
#[derive(Clone, Debug, Default)]
pub struct FlipHorizontal {
    pub keypoint_permutation: Option<Vec<usize>>,
}
impl Pass for FlipHorizontal {
    fn apply(&self, s: Sample) -> Result<Vec<Sample>> {
        let mut out = Warp {
            affine: Affine {
                a: -1.0,
                tx: s.image.width() as f32,
                ..Affine::IDENTITY
            },
            width: s.image.width(),
            height: s.image.height(),
            fill: [0; 3],
            interpolation: Interpolation::Nearest,
        }
        .transform(&s, "flip")?;
        if let Some(p) = &self.keypoint_permutation {
            if p.iter().enumerate().any(|(i, &j)| p.get(j) != Some(&i)) {
                return Err(invalid("keypoint permutation must be an involution"));
            }
            let mut objects = ObjectTable::default();
            for o in out.objects.iter() {
                let mut a = o.to_owned();
                if !a.keypoints.is_empty() {
                    if p.len() != a.keypoints.len() {
                        return Err(invalid("keypoint permutation length mismatch"));
                    }
                    a.keypoints = p.iter().map(|&i| a.keypoints[i]).collect();
                }
                objects.push(a)?;
            }
            out.objects = objects;
        }
        Ok(vec![out])
    }
}
/// One crop per matching instance. Padding is a fraction of each bbox dimension.
#[derive(Clone, Debug)]
pub struct InstanceCrops {
    pub classes: Option<Vec<u32>>,
    pub padding: f32,
    pub keep_neighbors: bool,
}
impl Default for InstanceCrops {
    fn default() -> Self {
        Self {
            classes: None,
            padding: 0.0,
            keep_neighbors: true,
        }
    }
}
impl Pass for InstanceCrops {
    fn apply(&self, s: Sample) -> Result<Vec<Sample>> {
        if !self.padding.is_finite() || self.padding < 0.0 {
            return Err(invalid("padding must be nonnegative"));
        }
        let mut out = vec![];
        for (index, o) in s.objects.iter().enumerate() {
            if self
                .classes
                .as_ref()
                .is_some_and(|c| !c.contains(&o.class_id()))
            {
                continue;
            }
            let b = o.bbox();
            let x = (b.x - b.width * self.padding).floor().max(0.0);
            let y = (b.y - b.height * self.padding).floor().max(0.0);
            let right = (b.right() + b.width * self.padding)
                .ceil()
                .min(s.image.width() as f32);
            let bottom = (b.bottom() + b.height * self.padding)
                .ceil()
                .min(s.image.height() as f32);
            if right <= x || bottom <= y {
                continue;
            }
            let mut crop = Crop::new(x as u32, y as u32, (right - x) as u32, (bottom - y) as u32)
                .transform_selected(
                &s,
                if self.keep_neighbors {
                    None
                } else {
                    Some(index)
                },
            )?;
            crop.provenance
                .last_mut()
                .expect("crop provenance")
                .parameters
                .insert("instance_index".into(), index.into());
            out.push(crop);
        }
        Ok(out)
    }
}

fn derived(
    s: &Sample,
    image: Raster,
    objects: ObjectTable,
    operation: &str,
    parameters: crate::Metadata,
) -> Sample {
    let uid = uuid::Uuid::new_v4();
    let mut provenance = s.provenance.clone();
    provenance.push(crate::ir::Provenance {
        parent_uid: s.uid,
        operation: operation.into(),
        parameters,
    });
    Sample {
        uid,
        id: format!("{}/{operation}/{uid}", s.id),
        image,
        objects,
        split: s.split.clone(),
        metadata: s.metadata.clone(),
        provenance,
    }
}

/// Extract each masked instance. Foreground RGB is unchanged; background is `fill`.
/// Object geometry and ALL keypoint slots are translated, not clipped or discarded.
/// Thus coordinates may lie outside the cropped image. Masks retain exact holes/islands.
#[derive(Clone, Debug, Default)]
pub struct MaskCrops {
    pub classes: Option<Vec<u32>>,
    /// Context padding in pixels, clipped to the source image.
    pub padding: u32,
    pub fill: [u8; 3],
}
impl MaskCrops {
    /// Associate an explicit image-sized IR mask with an object by table position.
    pub fn crop(&self, s: &Sample, object_index: usize, mask: &Mask) -> Result<Sample> {
        s.image.validate()?;
        s.objects.validate()?;
        self.crop_validated(s, object_index, mask)
    }
    pub fn crop_instance(&self, s: &Sample, object_index: usize) -> Result<Sample> {
        s.image.validate()?;
        s.objects.validate()?;
        let object = s
            .objects
            .get(object_index)
            .ok_or_else(|| invalid("object index out of range"))?;
        let mask = object
            .mask()
            .ok_or_else(|| invalid("instance has no mask"))?;
        self.crop_validated(s, object_index, mask)
    }
    fn crop_validated(&self, s: &Sample, object_index: usize, mask: &Mask) -> Result<Sample> {
        mask.validate()?;
        if (mask.width(), mask.height()) != (s.image.width(), s.image.height()) {
            return Err(invalid("mask dimensions do not match source image"));
        }
        let object = s
            .objects
            .get(object_index)
            .ok_or_else(|| invalid("object index out of range"))?;
        let (mask_x, mask_y, mask_width, mask_height) = mask
            .pixel_bounds()
            .ok_or_else(|| invalid("cannot crop an empty mask"))?;
        let x = mask_x.saturating_sub(self.padding);
        let y = mask_y.saturating_sub(self.padding);
        let right = (mask_x + mask_width)
            .saturating_add(self.padding)
            .min(s.image.width());
        let bottom = (mask_y + mask_height)
            .saturating_add(self.padding)
            .min(s.image.height());
        let (width, height) = (right - x, bottom - y);
        let mut pixels = vec![0; crate::ir::pixel_count(width, height)? * 3];
        pixels
            .par_chunks_mut(width as usize * 3)
            .enumerate()
            .for_each(|(dy, row)| {
                for (dx, out) in row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                    let (sx, sy) = (x + dx as u32, y + dy as u32);
                    if mask.get(sx, sy) {
                        let i = (sy as usize * s.image.width() as usize + sx as usize) * 3;
                        out.copy_from_slice(&s.image.pixels()[i..i + 3]);
                    } else {
                        *out = self.fill;
                    }
                }
            });
        let affine = Affine {
            tx: -(x as f32),
            ty: -(y as f32),
            ..Affine::IDENTITY
        };
        let mut annotation = object.to_owned();
        annotation.shape = affine.shape(&annotation.shape);
        annotation.bbox.x -= x as f32;
        annotation.bbox.y -= y as f32;
        for keypoint in &mut annotation.keypoints {
            keypoint.point = affine.map(keypoint.point);
        }
        annotation.mask = Some(Mask::from_fn(width, height, |dx, dy| {
            mask.get(x + dx, y + dy)
        })?);
        let mut objects = ObjectTable::default();
        objects.push(annotation)?;
        let parameters = crate::Metadata::from([
            ("origin".into(), serde_json::json!([x, y])),
            (
                "source_size".into(),
                serde_json::json!([s.image.width(), s.image.height()]),
            ),
            ("padding".into(), self.padding.into()),
            ("object_index".into(), object_index.into()),
            ("object_id".into(), object.id().into()),
            ("fill".into(), serde_json::json!(self.fill)),
        ]);
        Ok(derived(
            s,
            Raster::new(width, height, pixels)?,
            objects,
            "mask_crop",
            parameters,
        ))
    }
}
impl Pass for MaskCrops {
    /// Unmasked objects and empty masks produce no crops. Invalid masks return errors.
    fn apply(&self, s: Sample) -> Result<Vec<Sample>> {
        s.image.validate()?;
        s.objects.validate()?;
        let mut crops = Vec::new();
        for (i, object) in s.objects.iter().enumerate() {
            if self
                .classes
                .as_ref()
                .is_some_and(|classes| !classes.contains(&object.class_id()))
            {
                continue;
            }
            if let Some(mask) = object.mask() {
                if (mask.width(), mask.height()) != (s.image.width(), s.image.height()) {
                    return Err(invalid("mask dimensions do not match source image"));
                }
                if mask.area() != 0 {
                    crops.push(self.crop_validated(&s, i, mask)?);
                }
            }
        }
        Ok(crops)
    }
}
