#![forbid(unsafe_code)]
use cv_dataset_ir::{
    self as ir,
    packed::{self, wire},
    transforms::{MaskCrops, Pass},
};
use numpy::{
    IntoPyArray,
    ndarray::{ArrayD, IxDyn},
};
use pyo3::{
    exceptions::{PyIndexError, PyKeyError, PyOSError, PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyDict, PyList},
};
use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::PathBuf,
    sync::{Mutex, MutexGuard},
};

fn error(e: ir::Error) -> PyErr {
    match e {
        ir::Error::Io(e) => PyOSError::new_err(e.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}
fn lock<T>(mutex: &Mutex<T>) -> PyResult<MutexGuard<'_, T>> {
    mutex
        .lock()
        .map_err(|_| PyRuntimeError::new_err("reader/writer lock poisoned"))
}
fn json_value<'py>(py: Python<'py>, text: String) -> PyResult<Bound<'py, PyAny>> {
    py.import("json")?.getattr("loads")?.call1((text,))
}
fn to_json<T: serde::Serialize>(value: &T) -> PyResult<String> {
    serde_json::to_string(value).map_err(|e| PyValueError::new_err(e.to_string()))
}
fn array<'py>(py: Python<'py>, array: wire::Array) -> PyResult<Bound<'py, PyAny>> {
    let shape = IxDyn(&array.shape);
    macro_rules! numpy {
        ($values:expr) => {
            ArrayD::from_shape_vec(shape, $values)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .into_pyarray(py)
                .into_any()
        };
    }
    Ok(match array.dtype.as_str() {
        "|u1" => numpy!(array.data),
        "<u4" => numpy!(array.uint32()),
        "<u8" => numpy!(array.uint64()),
        "<f4" => numpy!(array.floats()),
        _ => return Err(PyValueError::new_err("unsupported dtype")),
    })
}
fn sample_to_py(py: Python<'_>, sample: ir::Sample) -> PyResult<Py<PyDict>> {
    let record = wire::Record::from_owned_sample(sample);
    let result = PyDict::new(py);
    result.set_item("uid", record.uid)?;
    result.set_item("id", record.id)?;
    result.set_item("split", json_value(py, to_json(&record.split)?)?)?;
    result.set_item("metadata", json_value(py, to_json(&record.metadata)?)?)?;
    result.set_item("provenance", json_value(py, to_json(&record.provenance)?)?)?;
    result.set_item("image", array(py, record.image)?)?;
    let objects = PyDict::new(py);
    let o = record.objects;
    for (name, a) in [
        ("ids", o.ids),
        ("class_ids", o.class_ids),
        ("boxes", o.boxes),
        ("shape_types", o.shape_types),
        ("shape_params", o.shape_params),
        ("polygon_object_offsets", o.polygon_object_offsets),
        ("polygon_offsets", o.polygon_offsets),
        ("vertices", o.vertices),
        ("keypoint_offsets", o.keypoint_offsets),
        ("keypoints", o.keypoints),
        ("visibility", o.visibility),
        ("is_crowd", o.is_crowd),
    ] {
        objects.set_item(name, array(py, a)?)?;
    }
    objects.set_item("metadata", json_value(py, to_json(&o.metadata)?)?)?;
    let masks = PyList::empty(py);
    for mask in o.masks {
        if let Some(mask) = mask {
            let item = PyDict::new(py);
            item.set_item("width", mask.width)?;
            item.set_item("height", mask.height)?;
            item.set_item("bitorder", mask.bitorder)?;
            item.set_item("data", mask.data.into_pyarray(py))?;
            masks.append(item)?;
        } else {
            masks.append(py.None())?;
        }
    }
    objects.set_item("masks", masks)?;
    result.set_item("objects", objects)?;
    Ok(result.unbind())
}
#[pyclass(name = "PackedReader")]
struct Reader {
    inner: Mutex<Option<packed::Reader<BufReader<File>>>>,
}
#[pymethods]
impl Reader {
    #[new]
    #[pyo3(signature=(path,max_record_bytes=536870912))]
    fn new(py: Python<'_>, path: PathBuf, max_record_bytes: u64) -> PyResult<Self> {
        let reader = py
            .detach(|| {
                packed::Reader::open(
                    path,
                    packed::Options {
                        max_record_bytes,
                        ..Default::default()
                    },
                )
            })
            .map_err(error)?;
        Ok(Self {
            inner: Mutex::new(Some(reader)),
        })
    }
    fn header_json(&self) -> PyResult<String> {
        let guard = lock(&self.inner)?;
        let reader = guard
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("reader is closed"))?;
        to_json(
            &serde_json::json!({"version":reader.version(),"categories":reader.categories(),"metadata":reader.metadata(),"sample_ids":reader.sample_ids().iter().map(ToString::to_string).collect::<Vec<_>>()}),
        )
    }
    fn read_index(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyDict>> {
        let sample = py.detach(|| {
            let mut guard = lock(&self.inner)?;
            let reader = guard
                .as_mut()
                .ok_or_else(|| PyValueError::new_err("reader is closed"))?;
            let uid = *reader
                .sample_ids()
                .get(index)
                .ok_or_else(|| PyIndexError::new_err(index))?;
            reader
                .read_sample(uid)
                .map_err(error)?
                .ok_or_else(|| PyKeyError::new_err(uid.to_string()))
        })?;
        sample_to_py(py, sample)
    }
    fn read_uid(&self, py: Python<'_>, uid: &str) -> PyResult<Py<PyDict>> {
        let uid = ir::Uuid::parse_str(uid).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let sample = py.detach(|| {
            let mut guard = lock(&self.inner)?;
            let reader = guard
                .as_mut()
                .ok_or_else(|| PyValueError::new_err("reader is closed"))?;
            reader
                .read_sample(uid)
                .map_err(error)?
                .ok_or_else(|| PyKeyError::new_err(uid.to_string()))
        })?;
        sample_to_py(py, sample)
    }
    fn close(&self) -> PyResult<()> {
        lock(&self.inner)?.take();
        Ok(())
    }
}
#[pyclass(name = "PackedWriter")]
struct Writer {
    inner: Mutex<Option<packed::StreamWriter<BufWriter<File>>>>,
}
#[pymethods]
impl Writer {
    #[new]
    #[pyo3(signature=(path,categories_json,metadata_json,samples,compression_level=3,max_record_bytes=536870912))]
    fn new(
        py: Python<'_>,
        path: PathBuf,
        categories_json: &str,
        metadata_json: &str,
        samples: u64,
        compression_level: i32,
        max_record_bytes: u64,
    ) -> PyResult<Self> {
        let categories: Vec<ir::Category> = serde_json::from_str(categories_json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let metadata: ir::Metadata = serde_json::from_str(metadata_json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let writer = py
            .detach(|| {
                ir::Dataset {
                    categories: categories.clone(),
                    ..Default::default()
                }
                .validate()?;
                let out = BufWriter::new(File::create_new(path)?);
                packed::StreamWriter::new(
                    out,
                    &categories,
                    &metadata,
                    samples,
                    packed::Options {
                        compression_level,
                        max_record_bytes,
                    },
                )
            })
            .map_err(error)?;
        Ok(Self {
            inner: Mutex::new(Some(writer)),
        })
    }
    fn push(&self, py: Python<'_>, record: Vec<u8>) -> PyResult<()> {
        py.detach(|| {
            let sample = packed::decode_record(&record).map_err(error)?;
            let mut guard = lock(&self.inner)?;
            guard
                .as_mut()
                .ok_or_else(|| PyValueError::new_err("writer is closed"))?
                .push(&sample)
                .map_err(error)
        })
    }
    fn finish(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let writer = lock(&self.inner)?
                .take()
                .ok_or_else(|| PyValueError::new_err("writer is closed"))?;
            writer.finish().map_err(error)?;
            Ok(())
        })
    }
    fn close(&self) -> PyResult<()> {
        lock(&self.inner)?.take();
        Ok(())
    }
}
#[pyfunction]
#[pyo3(signature=(record,classes=None,padding=0,fill=[0,0,0],object_index=None,mask=None))]
fn mask_crops(
    py: Python<'_>,
    record: Vec<u8>,
    classes: Option<Vec<u32>>,
    padding: u32,
    fill: [u8; 3],
    object_index: Option<usize>,
    mask: Option<Vec<u8>>,
) -> PyResult<Vec<Py<PyDict>>> {
    let result = py
        .detach(|| {
            let sample = packed::decode_record(&record)?;
            let pass = MaskCrops {
                classes,
                padding,
                fill,
            };
            if let Some(index) = object_index {
                let crop = if let Some(mask) = mask {
                    let mask =
                        ir::Mask::from_dense(sample.image.width(), sample.image.height(), &mask)?;
                    pass.crop(&sample, index, &mask)?
                } else {
                    pass.crop_instance(&sample, index)?
                };
                Ok(vec![crop])
            } else if mask.is_some() {
                Err(ir::Error::Invalid(
                    "an explicit mask needs an object index".into(),
                ))
            } else {
                pass.apply(sample)
            }
        })
        .map_err(error)?;
    result.into_iter().map(|s| sample_to_py(py, s)).collect()
}
#[pyfunction]
fn transform(
    py: Python<'_>,
    record: Vec<u8>,
    operation: &str,
    parameters_json: &str,
) -> PyResult<Py<PyDict>> {
    use ir::transforms::{Crop, FlipHorizontal, Resize, Rotate, Zoom};
    let params: serde_json::Value =
        serde_json::from_str(parameters_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = py
        .detach(|| {
            let sample = packed::decode_record(&record)?;
            let integer = |key: &str| {
                params[key]
                    .as_u64()
                    .and_then(|v| u32::try_from(v).ok())
                    .ok_or_else(|| ir::Error::Invalid(format!("invalid {key}")))
            };
            let number = |key: &str| {
                params[key]
                    .as_f64()
                    .map(|v| v as f32)
                    .ok_or_else(|| ir::Error::Invalid(format!("invalid {key}")))
            };
            let fill = || {
                serde_json::from_value::<[u8; 3]>(params["fill"].clone()).map_err(ir::Error::from)
            };
            match operation {
                "resize" => Resize {
                    width: integer("width")?,
                    height: integer("height")?,
                }
                .apply(sample),
                "crop" => Crop::new(
                    integer("x")?,
                    integer("y")?,
                    integer("width")?,
                    integer("height")?,
                )
                .apply(sample),
                "rotate" => Rotate {
                    degrees: number("degrees")?,
                    fill: fill()?,
                }
                .apply(sample),
                "zoom" => Zoom {
                    factor: number("factor")?,
                    fill: fill()?,
                }
                .apply(sample),
                "flip" => FlipHorizontal {
                    keypoint_permutation: serde_json::from_value(
                        params["keypoint_permutation"].clone(),
                    )?,
                }
                .apply(sample),
                _ => Err(ir::Error::Invalid("unknown transform".into())),
            }
        })
        .map_err(error)?;
    sample_to_py(
        py,
        result
            .into_iter()
            .next()
            .ok_or_else(|| PyValueError::new_err("transform produced no sample"))?,
    )
}
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Reader>()?;
    module.add_class::<Writer>()?;
    module.add_function(wrap_pyfunction!(mask_crops, module)?)?;
    module.add_function(wrap_pyfunction!(transform, module)?)?;
    Ok(())
}
