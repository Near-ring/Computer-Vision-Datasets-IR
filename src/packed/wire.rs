//! Language-neutral v2 wire schema. NumPy arrays use `{dtype, shape, data}` maps.
use crate::{
    Annotation, Keypoint, Mask, Metadata, ObjectTable, Point, Provenance, Raster, Rect, Result,
    Sample, Shape, Split, Uuid, Visibility, invalid,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Array {
    pub dtype: String,
    pub shape: Vec<usize>,
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}
impl Array {
    fn u8(values: Vec<u8>, shape: Vec<usize>) -> Self {
        Self {
            dtype: "|u1".into(),
            shape,
            data: values,
        }
    }
    fn u32(values: impl IntoIterator<Item = u32>, shape: Vec<usize>) -> Self {
        Self {
            dtype: "<u4".into(),
            shape,
            data: values.into_iter().flat_map(u32::to_le_bytes).collect(),
        }
    }
    fn u64(values: impl IntoIterator<Item = u64>, shape: Vec<usize>) -> Self {
        Self {
            dtype: "<u8".into(),
            shape,
            data: values.into_iter().flat_map(u64::to_le_bytes).collect(),
        }
    }
    fn f32(values: impl IntoIterator<Item = f32>, shape: Vec<usize>) -> Self {
        Self {
            dtype: "<f4".into(),
            shape,
            data: values.into_iter().flat_map(f32::to_le_bytes).collect(),
        }
    }
    pub fn validate(&self, dtype: &str, shape: &[usize]) -> Result<()> {
        let size = match dtype {
            "|u1" => 1,
            "<u4" | "<f4" => 4,
            "<u8" => 8,
            _ => return Err(invalid("unknown array dtype")),
        };
        let bytes = shape
            .iter()
            .try_fold(size, |n: usize, &v| n.checked_mul(v))
            .ok_or_else(|| invalid("wire array size overflow"))?;
        if self.dtype != dtype || self.shape != shape || self.data.len() != bytes {
            return Err(invalid(format!(
                "invalid wire array: expected {dtype} {shape:?}"
            )));
        }
        Ok(())
    }
    pub fn floats(&self) -> Vec<f32> {
        self.data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect()
    }
    pub fn uint32(&self) -> Vec<u32> {
        self.data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| u32::from_le_bytes(*v))
            .collect()
    }
    pub fn uint64(&self) -> Vec<u64> {
        self.data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|v| u64::from_le_bytes(*v))
            .collect()
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireSplit {
    pub kind: String,
    pub name: Option<String>,
}
impl From<&Split> for WireSplit {
    fn from(split: &Split) -> Self {
        Self {
            kind: match split {
                Split::Named(_) => "named",
                other => other.as_str(),
            }
            .into(),
            name: match split {
                Split::Named(name) => Some(name.clone()),
                _ => None,
            },
        }
    }
}
impl TryFrom<WireSplit> for Split {
    type Error = crate::Error;
    fn try_from(s: WireSplit) -> Result<Self> {
        match (s.kind.as_str(), s.name) {
            ("train", None) => Ok(Self::Train),
            ("val", None) => Ok(Self::Val),
            ("test", None) => Ok(Self::Test),
            ("unassigned", None) => Ok(Self::Unassigned),
            ("named", Some(name)) => Ok(Self::Named(name)),
            _ => Err(invalid("invalid wire split")),
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireMask {
    pub width: u32,
    pub height: u32,
    pub bitorder: String,
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireProvenance {
    pub parent_uid: String,
    pub operation: String,
    pub parameters: Metadata,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Objects {
    pub ids: Array,
    pub class_ids: Array,
    pub boxes: Array,
    pub shape_types: Array,
    pub shape_params: Array,
    pub polygon_object_offsets: Array,
    pub polygon_offsets: Array,
    pub vertices: Array,
    pub keypoint_offsets: Array,
    pub keypoints: Array,
    pub visibility: Array,
    pub is_crowd: Array,
    pub masks: Vec<Option<WireMask>>,
    pub metadata: Vec<Metadata>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub uid: String,
    pub id: String,
    pub split: WireSplit,
    pub image: Array,
    pub objects: Objects,
    pub metadata: Metadata,
    pub provenance: Vec<WireProvenance>,
}
impl Record {
    pub fn from_sample(s: &Sample) -> Self {
        let mut record = Self::without_pixels(s);
        record.image.data = s.image.pixels().to_vec();
        record
    }
    /// Move RGB ownership into a record, avoiding another full image copy.
    pub fn from_owned_sample(s: Sample) -> Self {
        let mut record = Self::without_pixels(&s);
        record.image.data = s.image.into_pixels();
        record
    }
    fn without_pixels(s: &Sample) -> Self {
        let mut ids = vec![];
        let mut classes = vec![];
        let mut boxes = vec![];
        let mut types = vec![];
        let mut params = vec![];
        let mut polygon_objects = vec![0];
        let mut polygons = vec![0];
        let mut vertices = vec![];
        let mut keypoint_offsets = vec![0];
        let mut keypoints = vec![];
        let mut visibility = vec![];
        let mut crowds = vec![];
        let mut masks = vec![];
        let mut metadata = vec![];
        for object in s.objects.iter() {
            ids.push(object.id());
            classes.push(object.class_id());
            let b = object.bbox();
            boxes.extend([b.x, b.y, b.width, b.height]);
            let (kind, values) = match object.shape() {
                Shape::Rect(r) => (0, [r.x, r.y, r.width, r.height]),
                Shape::Circle { center, radius } => (1, [center.x, center.y, radius, 0.0]),
                Shape::Polygons(parts) => {
                    for part in parts {
                        vertices.extend(part.iter().flat_map(|p| [p.x, p.y]));
                        polygons.push((vertices.len() / 2) as u64);
                    }
                    (2, [0.0; 4])
                }
                Shape::Point(p) => (3, [p.x, p.y, 0.0, 0.0]),
            };
            types.push(kind);
            params.extend(values);
            polygon_objects.push((polygons.len() - 1) as u64);
            for k in object.keypoints() {
                keypoints.extend([k.point.x, k.point.y]);
                visibility.push(k.visibility as u8);
            }
            keypoint_offsets.push(visibility.len() as u64);
            crowds.push(u8::from(object.is_crowd()));
            masks.push(object.mask().map(|m| WireMask {
                width: m.width(),
                height: m.height(),
                bitorder: "little".into(),
                data: m.bits().to_vec(),
            }));
            metadata.push(object.metadata().clone());
        }
        let n = ids.len();
        let v = vertices.len() / 2;
        let k = visibility.len();
        let rings = polygons.len() - 1;
        Self {
            uid: s.uid.to_string(),
            id: s.id.clone(),
            split: (&s.split).into(),
            image: Array::u8(
                vec![],
                vec![s.image.height() as usize, s.image.width() as usize, 3],
            ),
            objects: Objects {
                ids: Array::u64(ids, vec![n]),
                class_ids: Array::u32(classes, vec![n]),
                boxes: Array::f32(boxes, vec![n, 4]),
                shape_types: Array::u8(types, vec![n]),
                shape_params: Array::f32(params, vec![n, 4]),
                polygon_object_offsets: Array::u64(polygon_objects, vec![n + 1]),
                polygon_offsets: Array::u64(polygons, vec![rings + 1]),
                vertices: Array::f32(vertices, vec![v, 2]),
                keypoint_offsets: Array::u64(keypoint_offsets, vec![n + 1]),
                keypoints: Array::f32(keypoints, vec![k, 2]),
                visibility: Array::u8(visibility, vec![k]),
                is_crowd: Array::u8(crowds, vec![n]),
                masks,
                metadata,
            },
            metadata: s.metadata.clone(),
            provenance: s
                .provenance
                .iter()
                .map(|p| WireProvenance {
                    parent_uid: p.parent_uid.to_string(),
                    operation: p.operation.clone(),
                    parameters: p.parameters.clone(),
                })
                .collect(),
        }
    }
    pub fn into_sample(self) -> Result<Sample> {
        if self.image.shape.len() != 3 || self.image.shape[2] != 3 {
            return Err(invalid("image must be HWC RGB8"));
        }
        self.image.validate("|u1", &self.image.shape)?;
        let height =
            u32::try_from(self.image.shape[0]).map_err(|_| invalid("image height overflow"))?;
        let width =
            u32::try_from(self.image.shape[1]).map_err(|_| invalid("image width overflow"))?;
        let uid = Uuid::parse_str(&self.uid).map_err(|e| invalid(e.to_string()))?;
        if uid.is_nil() || self.id.is_empty() {
            return Err(invalid("invalid sample identity"));
        }
        let mut sample = Sample::new(self.id, Raster::new(width, height, self.image.data)?);
        sample.uid = uid;
        sample.split = self.split.try_into()?;
        sample.metadata = self.metadata;
        sample.provenance = self
            .provenance
            .into_iter()
            .map(|p| {
                Ok(Provenance {
                    parent_uid: Uuid::parse_str(&p.parent_uid)
                        .map_err(|e| invalid(e.to_string()))?,
                    operation: p.operation,
                    parameters: p.parameters,
                })
            })
            .collect::<Result<_>>()?;
        let o = self.objects;
        let n = *o
            .ids
            .shape
            .first()
            .ok_or_else(|| invalid("missing object dimension"))?;
        o.ids.validate("<u8", &[n])?;
        o.class_ids.validate("<u4", &[n])?;
        o.boxes.validate("<f4", &[n, 4])?;
        o.shape_types.validate("|u1", &[n])?;
        o.shape_params.validate("<f4", &[n, 4])?;
        o.is_crowd.validate("|u1", &[n])?;
        if o.metadata.len() != n || o.masks.len() != n {
            return Err(invalid("wire object column length mismatch"));
        }
        let vertices = check_matrix(&o.vertices, 2)?;
        let keypoints = check_matrix(&o.keypoints, 2)?;
        let rings = o
            .polygon_offsets
            .shape
            .first()
            .copied()
            .and_then(|v| v.checked_sub(1))
            .ok_or_else(|| invalid("missing polygon offset"))?;
        let object_offsets = offsets(&o.polygon_object_offsets, n, rings)?;
        let polygon_offsets = offsets(&o.polygon_offsets, rings, vertices.len() / 2)?;
        let keypoint_offsets = offsets(&o.keypoint_offsets, n, keypoints.len() / 2)?;
        o.visibility.validate("|u1", &[keypoints.len() / 2])?;
        let ids = o.ids.uint64();
        let classes = o.class_ids.uint32();
        let boxes = check_matrix(&o.boxes, 4)?;
        let params = check_matrix(&o.shape_params, 4)?;
        let mut objects = ObjectTable::default();
        for (i, (mask, metadata)) in o.masks.into_iter().zip(o.metadata).enumerate() {
            let p = &params[i * 4..i * 4 + 4];
            let shape = match o.shape_types.data[i] {
                0 => Shape::Rect(Rect::new(p[0], p[1], p[2], p[3])),
                1 => Shape::Circle {
                    center: Point::new(p[0], p[1]),
                    radius: p[2],
                },
                2 => Shape::Polygons(
                    (object_offsets[i]..object_offsets[i + 1])
                        .map(|ring| {
                            vertices[2 * polygon_offsets[ring]..2 * polygon_offsets[ring + 1]]
                                .as_chunks::<2>()
                                .0
                                .iter()
                                .map(|p| Point::new(p[0], p[1]))
                                .collect()
                        })
                        .collect(),
                ),
                3 => Shape::Point(Point::new(p[0], p[1])),
                _ => return Err(invalid("unknown shape type")),
            };
            if o.shape_types.data[i] != 2 && object_offsets[i] != object_offsets[i + 1] {
                return Err(invalid("non-polygon has polygon vertices"));
            }
            if o.is_crowd.data[i] > 1 {
                return Err(invalid("invalid crowd flag"));
            }
            let mut a = Annotation::new(classes[i], shape);
            a.id = ids[i];
            a.bbox = Rect::new(
                boxes[i * 4],
                boxes[i * 4 + 1],
                boxes[i * 4 + 2],
                boxes[i * 4 + 3],
            );
            a.is_crowd = o.is_crowd.data[i] != 0;
            a.metadata = metadata;
            for k in keypoint_offsets[i]..keypoint_offsets[i + 1] {
                a.keypoints.push(Keypoint::new(
                    keypoints[k * 2],
                    keypoints[k * 2 + 1],
                    Visibility::try_from(o.visibility.data[k])?,
                ));
            }
            a.mask = mask
                .map(|m| {
                    if m.bitorder != "little" || (m.width, m.height) != (width, height) {
                        return Err(invalid("invalid mask layout or dimensions"));
                    }
                    Mask::from_bits(m.width, m.height, m.data)
                })
                .transpose()?;
            objects.push(a)?;
        }
        sample.objects = objects;
        Ok(sample)
    }
}
fn check_matrix(array: &Array, columns: usize) -> Result<Vec<f32>> {
    if array.shape.len() != 2 || array.shape[1] != columns {
        return Err(invalid("invalid matrix shape"));
    }
    array.validate("<f4", &array.shape)?;
    let v = array.floats();
    if v.iter().any(|x| !x.is_finite()) {
        return Err(invalid("non-finite geometry"));
    }
    Ok(v)
}
fn offsets(array: &Array, rows: usize, end: usize) -> Result<Vec<usize>> {
    let len = rows
        .checked_add(1)
        .ok_or_else(|| invalid("offset count overflow"))?;
    array.validate("<u8", &[len])?;
    let values: Vec<usize> = array
        .uint64()
        .into_iter()
        .map(|v| usize::try_from(v).map_err(|_| invalid("offset overflow")))
        .collect::<Result<_>>()?;
    if values.first() != Some(&0)
        || values.last() != Some(&end)
        || values.windows(2).any(|w| w[0] > w[1])
    {
        return Err(invalid("invalid ragged array offsets"));
    }
    Ok(values)
}
