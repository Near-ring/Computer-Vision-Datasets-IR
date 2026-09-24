//! Run against locally downloaded COCO8, COCO8-pose, and COCO8-seg archives.
//! cargo run --release --example public_smoke -- target/public-data target/public-smoke
use cv_dataset_ir::{
    formats::{coco, labelme, yolo},
    packed,
    transforms::{Pipeline, Resize, Rotate},
    *,
};
use std::{collections::BTreeMap, io::Write, path::PathBuf, time::Instant};
fn compare(a: &Dataset, b: &Dataset) {
    assert_eq!(a.samples.len(), b.samples.len());
    let index = b.index().unwrap();
    for sample in &a.samples {
        let other = index.get(sample.uid).expect("UUID roundtrip");
        assert_eq!(sample.image, other.image);
        assert_eq!(sample.split, other.split);
        assert_eq!(sample.objects.len(), other.objects.len());
        for (x, y) in sample.objects.iter().zip(other.objects.iter()) {
            assert_eq!(x.class_id(), y.class_id());
            assert!((x.bbox().x - y.bbox().x).abs() < 0.001);
            assert!((x.bbox().area() - y.bbox().area()).abs() < 1.0);
            assert_eq!(x.keypoints(), y.keypoints());
        }
    }
}
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let base = PathBuf::from(args.next().unwrap_or_else(|| "target/public-data".into()));
    let output = PathBuf::from(args.next().unwrap_or_else(|| "target/public-smoke".into()));
    std::fs::create_dir(&output)?;
    let mut results = vec![];
    for (name, task) in [
        ("coco8", yolo::Task::Detection),
        ("coco8-pose", yolo::Task::Pose),
        ("coco8-seg", yolo::Task::Segmentation),
    ] {
        let start = Instant::now();
        let dataset = yolo::read(base.join(format!("{name}.yaml")), task)?;
        let read_ms = start.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(dataset.samples.len(), 8);
        let root = output.join(name);
        std::fs::create_dir(&root)?;
        packed::write(&dataset, root.join("dataset.cvir"), Default::default())?;
        assert_eq!(packed::read(root.join("dataset.cvir"))?, dataset);
        coco::write(&dataset, root.join("coco"), Default::default())?;
        compare(&dataset, &coco::read_dir(root.join("coco"))?);
        labelme::write(&dataset, root.join("labelme"), Default::default())?;
        compare(&dataset, &labelme::read(root.join("labelme"), None)?);
        yolo::write(&dataset, root.join("yolo"), task, Default::default())?;
        let yolo_roundtrip = yolo::read(root.join("yolo/data.yaml"), task)?;
        // Text coordinates incur decimal roundoff, including pose coordinates.
        assert_eq!(dataset.samples.len(), yolo_roundtrip.samples.len());
        for s in &dataset.samples {
            let index = yolo_roundtrip.index()?;
            let other = index.get(s.uid).unwrap();
            assert_eq!(s.split, other.split);
            assert_eq!(s.objects.len(), other.objects.len());
            for (a, b) in s.objects.iter().zip(other.objects.iter()) {
                for (a, b) in a.keypoints().iter().zip(b.keypoints()) {
                    assert!(
                        (a.point.x - b.point.x).abs() < 0.001
                            && (a.point.y - b.point.y).abs() < 0.001
                    );
                    assert_eq!(a.visibility, b.visibility);
                }
            }
        }
        let start = Instant::now();
        let transformed = Pipeline::new()
            .then(Resize {
                width: 320,
                height: 320,
            })
            .then(Rotate {
                degrees: 12.0,
                fill: [0; 3],
            })
            .run(dataset.clone())?;
        let transform_ms = start.elapsed().as_secs_f64() * 1000.0;
        transformed.validate()?;
        let mut splits = BTreeMap::new();
        for s in &dataset.samples {
            *splits.entry(s.split.as_str()).or_insert(0) += 1;
        }
        let entry = serde_json::json!({"dataset":name,"samples":dataset.samples.len(),"annotations":dataset.samples.iter().map(|s|s.objects.len()).sum::<usize>(),"splits":splits,"read_ms":read_ms,"resize_rotate_ms":transform_ms,"roundtrips":["packed","coco","labelme","yolo"],"packed_bytes":std::fs::metadata(root.join("dataset.cvir"))?.len()});
        println!("{entry}");
        results.push(entry);
    }
    let report = serde_json::json!({"simd":format!("{:?}",fast_image_resize::Resizer::new().cpu_extensions()),"rayon_threads":rayon::current_num_threads(),"results":results});
    let mut out = std::io::BufWriter::new(std::fs::File::create_new(output.join("report.json"))?);
    serde_json::to_writer_pretty(&mut out, &report)?;
    out.flush()?;
    Ok(())
}
