mod support;
use cv_dataset_ir::{
    formats::{self, ExportOptions, LossPolicy, SplitPolicy, coco, labelme, yolo},
    *,
};
use std::fs;
#[test]
fn coco_polygon_pose_and_split_roundtrip() {
    let d = support::fixture();
    let t = tempfile::tempdir().unwrap();
    let root = t.path().join("coco");
    let report = coco::write(&d, &root, Default::default()).unwrap();
    assert_eq!(report.samples, 3);
    assert_eq!(report.annotations, 6);
    let out = coco::read_dir(root).unwrap();
    support::assert_identity(&d, &out);
    assert_eq!(d.metadata, out.metadata);
    for s in &d.samples {
        let index = out.index().unwrap();
        let o = index.get(s.uid).unwrap();
        for (a, b) in s.objects.iter().zip(o.objects.iter()) {
            assert_eq!(a.shape(), b.shape());
            assert_eq!(a.keypoints(), b.keypoints());
        }
    }
}
#[test]
fn labelme_primitives_pose_and_embedded_masks() {
    let mut d = support::fixture();
    let s = &mut d.samples[0];
    s.objects
        .push(Annotation::new(
            1,
            Shape::Circle {
                center: Point::new(6.0, 6.0),
                radius: 2.0,
            },
        ))
        .unwrap();
    s.objects
        .push(Annotation::new(1, Shape::Point(Point::new(12.0, 4.0))))
        .unwrap();
    let mut a = Annotation::new(1, Shape::Rect(Rect::new(1.0, 1.0, 10.0, 9.0)));
    a.mask = Some(
        Mask::from_fn(16, 12, |x, y| {
            (1..11).contains(&x) && (1..10).contains(&y) && !(4..7).contains(&x)
        })
        .unwrap(),
    );
    s.objects.push(a).unwrap();
    let t = tempfile::tempdir().unwrap();
    let root = t.path().join("lm");
    labelme::write(&d, &root, Default::default()).unwrap();
    let out = labelme::read(root, None).unwrap();
    support::assert_identity(&d, &out);
    for s in &d.samples {
        let index = out.index().unwrap();
        let b = index.get(s.uid).unwrap();
        for (a, b) in s.objects.iter().zip(b.objects.iter()) {
            assert_eq!(a.shape(), b.shape());
            assert_eq!(a.keypoints(), b.keypoints());
            assert_eq!(a.mask(), b.mask());
        }
    }
}
#[test]
fn yolo_all_tasks_three_images_each() {
    let t = tempfile::tempdir().unwrap();
    for (name, task, d) in [
        ("detect", yolo::Task::Detection, support::detection()),
        ("pose", yolo::Task::Pose, support::pose()),
        ("seg", yolo::Task::Segmentation, support::segmentation()),
    ] {
        let root = t.path().join(name);
        yolo::write(&d, &root, task, Default::default()).unwrap();
        let out = yolo::read(root.join("data.yaml"), task).unwrap();
        support::assert_identity(&d, &out);
        for s in &d.samples {
            let index = out.index().unwrap();
            let b = index.get(s.uid).unwrap();
            for (a, b) in s.objects.iter().zip(b.objects.iter()) {
                assert_eq!(a.class_id(), b.class_id());
                assert!((a.bbox().area() - b.bbox().area()).abs() < 1e-4);
                assert_eq!(a.keypoints(), b.keypoints());
            }
        }
    }
}
#[test]
fn ignore_splits_is_roundtrippable() {
    let d = support::detection();
    let t = tempfile::tempdir().unwrap();
    let options = ExportOptions {
        splits: SplitPolicy::Ignore,
        ..Default::default()
    };
    let root = t.path().join("y");
    yolo::write(&d, &root, yolo::Task::Detection, options).unwrap();
    let out = yolo::read(root.join("data.yaml"), yolo::Task::Detection).unwrap();
    assert!(out.samples.iter().all(|s| s.split == Split::Unassigned));
    assert_eq!(out.samples.len(), 3);
    let root = t.path().join("c");
    coco::write(&d, &root, options).unwrap();
    assert!(
        coco::read_dir(root)
            .unwrap()
            .samples
            .iter()
            .all(|s| s.split == Split::Unassigned)
    );
}
#[test]
fn loss_is_explicit_and_output_not_created_on_rejection() {
    let d = support::fixture();
    let t = tempfile::tempdir().unwrap();
    let root = t.path().join("reject");
    assert!(yolo::write(&d, &root, yolo::Task::Detection, Default::default()).is_err());
    assert!(!root.exists());
    let report = yolo::write(
        &d,
        &root,
        yolo::Task::Detection,
        ExportOptions {
            loss: LossPolicy::Allow,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!report.warnings.is_empty());
    assert!(yolo::write(&d, &root, yolo::Task::Detection, Default::default()).is_err());
}
#[test]
fn coco_uncompressed_and_compressed_mask_holes() {
    let t = tempfile::tempdir().unwrap();
    Raster::new(3, 2, vec![0; 18])
        .unwrap()
        .write_png(t.path().join("i.png"))
        .unwrap();
    let doc = serde_json::json!({"images":[{"id":100,"file_name":"i.png","width":3,"height":2}],"categories":[{"id":91,"name":"hole"}],"annotations":[{"id":77,"image_id":100,"category_id":91,"bbox":[0,0,3,2],"iscrowd":1,"segmentation":{"size":[2,3],"counts":[0,3,1,2]}}]});
    fs::write(t.path().join("a.json"), serde_json::to_vec(&doc).unwrap()).unwrap();
    let d = coco::read(&[coco::Source {
        annotations: t.path().join("a.json"),
        images: t.path().to_path_buf(),
        split: Split::Train,
    }])
    .unwrap();
    let m = d.samples[0].objects.get(0).unwrap().mask().unwrap();
    assert_eq!(m.to_dense(), [1, 1, 1, 1, 0, 1]);
    assert_eq!(d.samples[0].objects.class_ids(), [0]);
    let root = t.path().join("roundtrip");
    coco::write(&d, &root, Default::default()).unwrap();
    let out = coco::read_dir(root).unwrap();
    assert_eq!(out.samples[0].objects.get(0).unwrap().mask(), Some(m));
    assert!(out.samples[0].objects.get(0).unwrap().is_crowd());
}
#[test]
fn external_labelme_embedded_image_circle_and_split() {
    use base64::Engine;
    let t = tempfile::tempdir().unwrap();
    fs::create_dir(t.path().join("val")).unwrap();
    let image = Raster::new(8, 8, vec![12; 8 * 8 * 3]).unwrap();
    image.write_png(t.path().join("im.png")).unwrap();
    let data = base64::engine::general_purpose::STANDARD
        .encode(fs::read(t.path().join("im.png")).unwrap());
    let doc = serde_json::json!({"imageWidth":8,"imageHeight":8,"imageData":data,"shapes":[{"label":"ball","shape_type":"circle","points":[[4,4],[6,4]]}]});
    fs::write(
        t.path().join("val/a.json"),
        serde_json::to_vec(&doc).unwrap(),
    )
    .unwrap();
    let d = labelme::read(t.path(), None).unwrap();
    assert_eq!(d.samples[0].split, Split::Val);
    assert_eq!(d.samples[0].image, image);
    assert_eq!(
        d.samples[0].objects.get(0).unwrap().shape(),
        Shape::Circle {
            center: Point::new(4.0, 4.0),
            radius: 2.0
        }
    );
}
#[test]
fn external_yolo_lists_backgrounds_and_two_dimensional_pose() {
    let t = tempfile::tempdir().unwrap();
    fs::create_dir_all(t.path().join("images/train")).unwrap();
    fs::create_dir_all(t.path().join("labels/train")).unwrap();
    for i in 0..3 {
        Raster::new(10, 10, vec![0; 300])
            .unwrap()
            .write_png(t.path().join(format!("images/train/{i}.png")))
            .unwrap();
    }
    fs::write(
        t.path().join("labels/train/0.txt"),
        "0 0.5 0.5 0.4 0.4 0.2 0.3\n",
    )
    .unwrap();
    fs::write(
        t.path().join("list.txt"),
        "./images/train/0.png\n./images/train/1.png\n./images/train/2.png\n",
    )
    .unwrap();
    fs::write(
        t.path().join("data.yaml"),
        "names: [plant]\ntrain: list.txt\nkpt_shape: [1, 2]\n",
    )
    .unwrap();
    let d = yolo::read(t.path().join("data.yaml"), yolo::Task::Pose).unwrap();
    assert_eq!(d.samples.len(), 3);
    assert_eq!(d.samples.iter().map(|s| s.objects.len()).sum::<usize>(), 1);
    assert_eq!(
        d.samples[0].objects.get(0).unwrap().keypoints()[0],
        Keypoint::new(2.0, 3.0, Visibility::Visible)
    );
    fs::write(
        t.path().join("labels/train/0.txt"),
        "0 NaN 0.5 0.4 0.4 0.2 0.3",
    )
    .unwrap();
    assert!(yolo::read(t.path().join("data.yaml"), yolo::Task::Pose).is_err());
}
#[test]
fn path_traversal_and_invalid_metadata_are_rejected() {
    let mut d = support::detection();
    d.samples[0].split = Split::Named("../escape".into());
    let t = tempfile::tempdir().unwrap();
    assert!(labelme::write(&d, t.path().join("out"), Default::default()).is_err());
    let doc = serde_json::json!({"images":[{"id":1,"file_name":"../bad.png","width":2,"height":2}],"categories":[],"annotations":[]});
    fs::write(t.path().join("a.json"), doc.to_string()).unwrap();
    assert!(
        coco::read(&[coco::Source {
            annotations: t.path().join("a.json"),
            images: t.path().to_path_buf(),
            split: Split::Train
        }])
        .is_err()
    );
}
#[test]
fn uniform_frontend_backend_api() {
    let t = tempfile::tempdir().unwrap();
    let root = t.path().join("lm");
    let d = support::detection();
    let backend = labelme::Writer {
        directory: root.clone(),
        options: Default::default(),
    };
    let b: &dyn formats::Backend = &backend;
    b.write(&d).unwrap();
    let frontend = labelme::Reader {
        directory: root,
        categories: None,
    };
    let f: &dyn formats::Frontend = &frontend;
    support::assert_identity(&d, &f.read().unwrap());
}

#[test]
fn all_absent_labelme_pose_and_reserved_metadata() {
    let mut d = support::pose();
    for sample in &mut d.samples {
        let mut a = sample.objects.get(0).unwrap().to_owned();
        a.keypoints
            .fill(Keypoint::new(0.0, 0.0, Visibility::Absent));
        sample.objects = ObjectTable::default();
        sample.objects.push(a).unwrap();
    }
    let t = tempfile::tempdir().unwrap();
    let root = t.path().join("lm");
    labelme::write(&d, &root, Default::default()).unwrap();
    let result = labelme::read(root, None).unwrap();
    for sample in result.samples {
        assert_eq!(
            sample.objects.get(0).unwrap().keypoints(),
            vec![Keypoint::new(0.0, 0.0, Visibility::Absent); 2]
        );
    }
    d.categories[0]
        .metadata
        .insert("name".into(), "collision".into());
    assert!(coco::write(&d, t.path().join("bad"), Default::default()).is_err());
    assert!(!t.path().join("bad").exists());
}

#[test]
fn mask_to_yolo_reports_contours_holes_and_components() {
    let mut d = support::detection();
    for s in &mut d.samples {
        let mut a = Annotation::new(0, Shape::Rect(Rect::new(1.0, 1.0, 14.0, 10.0)));
        a.mask = Some(
            Mask::from_fn(16, 12, |x, y| {
                (1..7).contains(&x) && (1..10).contains(&y) && !(x == 3 && y == 4)
                    || (11..15).contains(&x) && (2..9).contains(&y)
            })
            .unwrap(),
        );
        s.objects = ObjectTable::default();
        s.objects.push(a).unwrap();
    }
    let t = tempfile::tempdir().unwrap();
    let root = t.path().join("yolo");
    assert!(yolo::write(&d, &root, yolo::Task::Segmentation, Default::default()).is_err());
    let report = yolo::write(
        &d,
        &root,
        yolo::Task::Segmentation,
        ExportOptions {
            loss: LossPolicy::Allow,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(report.warnings.iter().any(|s| s.contains("holes")));
    assert!(report.warnings.iter().any(|s| s.contains("largest")));
    assert_eq!(
        yolo::read(root.join("data.yaml"), yolo::Task::Segmentation)
            .unwrap()
            .samples
            .len(),
        3
    );
}

#[test]
fn empty_datasets_keep_vocabulary_and_metadata() {
    let mut d = support::detection();
    d.samples.clear();
    let t = tempfile::tempdir().unwrap();
    coco::write(&d, t.path().join("c"), Default::default()).unwrap();
    let out = coco::read_dir(t.path().join("c")).unwrap();
    assert_eq!(out.categories.len(), 2);
    assert_eq!(out.metadata, d.metadata);
    labelme::write(&d, t.path().join("l"), Default::default()).unwrap();
    assert_eq!(labelme::read(t.path().join("l"), None).unwrap(), d);
    yolo::write(
        &d,
        t.path().join("y"),
        yolo::Task::Detection,
        Default::default(),
    )
    .unwrap();
    assert_eq!(
        yolo::read(t.path().join("y/data.yaml"), yolo::Task::Detection).unwrap(),
        d
    );
}

#[test]
fn labelme_fractional_masks_and_underscore_filenames() {
    use base64::Engine;
    use image::ImageEncoder;
    let t = tempfile::tempdir().unwrap();
    let mut bytes = vec![];
    image::codecs::png::PngEncoder::new(&mut bytes)
        .write_image(&[255; 6], 3, 2, image::ExtendedColorType::L8)
        .unwrap();
    let bitmap = base64::engine::general_purpose::STANDARD.encode(bytes);
    Raster::new(8, 8, vec![0; 8 * 8 * 3])
        .unwrap()
        .write_png(t.path().join("image.png"))
        .unwrap();
    let doc = serde_json::json!({"imageWidth":8,"imageHeight":8,"imagePath":"image.png","shapes":[
        {"label":"mask","shape_type":"mask","points":[[1.5,1.5],[3.5,2.5]],"mask":bitmap},
        {"label":"mask","shape_type":"mask","points":[[-1.2,5.2],[0.8,6.2]],"mask":bitmap}
    ]});
    fs::write(t.path().join("_sample.json"), doc.to_string()).unwrap();
    let dataset = labelme::read(t.path(), None).unwrap();
    assert_eq!(dataset.samples.len(), 1);
    let first = dataset.samples[0].objects.get(0).unwrap().mask().unwrap();
    assert_eq!(first.area(), 6);
    assert_eq!(first.bounds().unwrap(), Rect::new(2.0, 2.0, 3.0, 2.0));
    let clipped = dataset.samples[0].objects.get(1).unwrap().mask().unwrap();
    assert_eq!(clipped.area(), 4);
    assert_eq!(clipped.bounds().unwrap(), Rect::new(0.0, 5.0, 2.0, 2.0));
}
