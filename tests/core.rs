mod support;
use cv_dataset_ir::{packed, transforms::*, *};
use std::io::Cursor;
#[test]
fn uuid_index_and_packed_random_access() {
    let d = support::fixture();
    let uid = d.samples[1].uid;
    assert_eq!(
        d.index().unwrap().get_u128(uid.as_u128()).unwrap().id,
        "frame-1"
    );
    let mut bytes = vec![];
    packed::write_to(&d, &mut bytes, packed::Options::default()).unwrap();
    let mut reader = packed::Reader::new(Cursor::new(bytes), packed::Options::default()).unwrap();
    assert_eq!(reader.sample_ids().len(), 3);
    assert_eq!(reader.read_sample(uid).unwrap().unwrap(), d.samples[1]);
    assert!(reader.read_sample(Uuid::new_v4()).unwrap().is_none());
    assert_eq!(reader.read_dataset().unwrap(), d);
}
#[test]
fn pack_corruption_limits_and_duplicate_ids() {
    let d = support::fixture();
    let mut bytes = vec![];
    packed::write_to(&d, &mut bytes, packed::Options::default()).unwrap();
    for n in [0, 7, 12, bytes.len() - 1] {
        assert!(
            packed::Reader::new(Cursor::new(bytes[..n].to_vec()), packed::Options::default())
                .is_err()
        );
    }
    bytes[0] = 0;
    assert!(packed::Reader::new(Cursor::new(bytes), packed::Options::default()).is_err());
    assert!(
        packed::write_to(
            &d,
            vec![],
            packed::Options {
                max_record_bytes: 100,
                ..Default::default()
            }
        )
        .is_err()
    );
    let mut d = d;
    d.samples[1].uid = d.samples[0].uid;
    assert!(d.validate().is_err());
    assert!(d.index().is_err());
}
#[test]
fn pack_does_not_overwrite() {
    let t = tempfile::tempdir().unwrap();
    let p = t.path().join("test.cvir");
    let d = support::fixture();
    packed::write(&d, &p, Default::default()).unwrap();
    assert!(packed::write(&d, &p, Default::default()).is_err());
    assert_eq!(packed::read(p).unwrap(), d);
}
#[test]
fn mask_bitpacking_roundtrip() {
    let dense = [0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1];
    let m = Mask::from_dense(5, 3, &dense).unwrap();
    assert_eq!(m.to_dense(), dense);
    assert_eq!(m.area(), 7);
    m.validate().unwrap();
    assert!(!m.get(5, 0));
    assert!(Mask::from_dense(4, 4, &dense).is_err());
}
#[test]
fn crop_updates_pixels_pose_mask_and_provenance() {
    let mut d = support::pose();
    let s = &mut d.samples[0];
    let mut a = s.objects.get(0).unwrap().to_owned();
    a.mask = Some(
        Mask::from_fn(16, 12, |x, y| {
            (2..10).contains(&x) && (2..8).contains(&y) && !(x == 4 && y == 4)
        })
        .unwrap(),
    );
    s.objects = ObjectTable::default();
    s.objects.push(a).unwrap();
    let out = Crop::new(4, 2, 8, 8).transform(s).unwrap();
    assert_ne!(out.uid, s.uid);
    assert_eq!(out.split, s.split);
    assert_eq!(out.provenance.last().unwrap().parent_uid, s.uid);
    assert_eq!(out.metadata, s.metadata);
    assert_eq!(
        &out.image.pixels()[..3],
        &s.image.pixels()[(2 * 16 + 4) * 3..(2 * 16 + 4) * 3 + 3]
    );
    let o = out.objects.get(0).unwrap();
    assert_eq!(o.bbox(), Rect::new(0.0, 0.0, 6.0, 6.0));
    assert_eq!(o.keypoints()[0].visibility, Visibility::Absent);
    assert_eq!(o.keypoints()[1].point, Point::new(5.0, 4.0));
    assert_eq!(o.mask().unwrap().area(), 35);
    assert!(!o.mask().unwrap().get(0, 2));
}
#[test]
fn concave_crop_keeps_disconnected_components() {
    let shape = Shape::Polygons(vec![vec![
        Point::new(0.0, 0.0),
        Point::new(6.0, 0.0),
        Point::new(6.0, 6.0),
        Point::new(4.0, 6.0),
        Point::new(4.0, 2.0),
        Point::new(2.0, 2.0),
        Point::new(2.0, 6.0),
        Point::new(0.0, 6.0),
    ]]);
    let clipped = shape.clip(Rect::new(0.0, 3.0, 6.0, 3.0)).unwrap();
    let Shape::Polygons(parts) = clipped else {
        panic!()
    };
    assert_eq!(parts.len(), 2);
    assert!((Shape::Polygons(parts).area() - 12.0).abs() < 1e-5);
}
#[test]
fn resize_scales_geometry_and_uses_nearest_masks() {
    let s = support::pose().samples.remove(0);
    let out = Resize {
        width: 32,
        height: 24,
    }
    .apply(s)
    .unwrap()
    .remove(0);
    assert_eq!(
        out.objects.get(0).unwrap().bbox(),
        Rect::new(4.0, 4.0, 16.0, 12.0)
    );
    assert_eq!(
        out.objects.get(0).unwrap().keypoints()[1].point,
        Point::new(18.0, 12.0)
    );
}
#[test]
fn rotate_and_flip_exact_pixels_and_pose_slots() {
    let image = Raster::new(2, 2, vec![1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0]).unwrap();
    let s = Sample::new("tiny", image);
    let out = Rotate {
        degrees: 90.0,
        fill: [0; 3],
    }
    .apply(s.clone())
    .unwrap()
    .remove(0);
    assert_eq!(out.image.pixels(), [3, 0, 0, 1, 0, 0, 4, 0, 0, 2, 0, 0]);
    let out = FlipHorizontal::default().apply(s).unwrap().remove(0);
    assert_eq!(out.image.pixels(), [2, 0, 0, 1, 0, 0, 4, 0, 0, 3, 0, 0]);
    let s = support::pose().samples.remove(0);
    let out = FlipHorizontal {
        keypoint_permutation: Some(vec![1, 0]),
    }
    .apply(s)
    .unwrap()
    .remove(0);
    let k = out.objects.get(0).unwrap().keypoints();
    assert_eq!(k[0].point, Point::new(7.0, 6.0));
    assert_eq!(k[0].visibility, Visibility::Occluded);
}
#[test]
fn affine_inverse_circle_and_zoom() {
    let a = Affine {
        a: 2.0,
        b: 0.3,
        c: 0.0,
        d: 3.0,
        tx: 12.0,
        ty: -5.0,
    };
    let p = Point::new(10.0, 20.0);
    let q = a.inverse().unwrap().map(a.map(p));
    assert!((q.x - p.x).abs() < 1e-4 && (q.y - p.y).abs() < 1e-4);
    assert!(matches!(
        a.shape(&Shape::Circle {
            center: p,
            radius: 2.0
        }),
        Shape::Polygons(_)
    ));
    let s = support::fixture().samples.remove(0);
    let o = Zoom {
        factor: 0.5,
        fill: [13, 14, 15],
    }
    .apply(s)
    .unwrap()
    .remove(0);
    assert_eq!(&o.image.pixels()[..3], &[13, 14, 15]);
    assert!(
        Zoom {
            factor: 0.0,
            fill: [0; 3]
        }
        .apply(o)
        .is_err()
    );
}
#[test]
fn stable_filter_and_instance_crops() {
    let d = support::fixture();
    let out = Pipeline::new()
        .then(Filter {
            classes: Some(vec![1]),
            ..Default::default()
        })
        .run(d.clone())
        .unwrap();
    for (a, b) in d.samples.iter().zip(&out.samples) {
        assert_eq!(a.uid, b.uid);
        assert_eq!(b.objects.class_ids(), [1]);
        assert_eq!(b.objects.vertices().len(), 3);
    }
    let out = Pipeline::new()
        .then(InstanceCrops {
            keep_neighbors: false,
            ..Default::default()
        })
        .run(d)
        .unwrap();
    assert_eq!(out.samples.len(), 6);
    for s in out.samples {
        assert_eq!(s.objects.len(), 1);
    }
}
#[test]
fn thread_counts_give_same_pixels_and_geometry() {
    let d = support::fixture();
    let p = Pipeline::new().then(Resize {
        width: 8,
        height: 6,
    });
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| p.run(d.clone()).unwrap())
    };
    let a = run(1);
    let b = run(4);
    for (a, b) in a.samples.iter().zip(&b.samples) {
        assert_eq!(a.image, b.image);
        assert_eq!(a.objects, b.objects);
        assert_eq!(a.split, b.split);
    }
}
#[test]
fn boundary_validation_and_filter_predicate() {
    assert!(Raster::new(0, 1, vec![]).is_err());
    assert!(Raster::new(1, 1, vec![0; 2]).is_err());
    assert!(
        Shape::Circle {
            center: Point::new(0.0, 0.0),
            radius: f32::NAN
        }
        .validate()
        .is_err()
    );
    let mut d = support::fixture();
    let out = Pipeline::new()
        .then(FilterObjects::new(|o: ObjectRef<'_>| {
            o.class_id() == 0 && o.area() > 10.0
        }))
        .run(d.clone())
        .unwrap();
    assert_eq!(out.samples[0].objects.len(), 1);
    d.ignore_splits();
    assert!(d.samples.iter().all(|s| s.split == Split::Unassigned));
    d.retain_split(&Split::Train);
    assert!(d.samples.is_empty());
}

#[test]
fn streaming_pack_and_checksum() {
    let dataset = support::fixture();
    let mut writer = packed::StreamWriter::new(
        Vec::new(),
        &dataset.categories,
        &dataset.metadata,
        3,
        Default::default(),
    )
    .unwrap();
    for sample in &dataset.samples {
        writer.push(sample).unwrap();
    }
    assert!(writer.push(&dataset.samples[0]).is_err());
    let mut bytes = writer.finish().unwrap();
    assert_eq!(
        packed::Reader::new(Cursor::new(&bytes), Default::default())
            .unwrap()
            .read_dataset()
            .unwrap(),
        dataset
    );
    // zstd frame checksum is at the end; damage it while retaining valid framing.
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    let mut reader = packed::Reader::new(Cursor::new(bytes), Default::default()).unwrap();
    assert!(reader.read_sample(dataset.samples[2].uid).is_err());
    let writer = packed::StreamWriter::new(
        Vec::new(),
        &dataset.categories,
        &dataset.metadata,
        3,
        Default::default(),
    )
    .unwrap();
    assert!(writer.finish().is_err());
}

#[test]
fn edge_points_survive_flip_and_mask_selection_moves_storage() {
    let mut sample = Sample::new("edge", Raster::new(8, 8, vec![0; 8 * 8 * 3]).unwrap());
    sample
        .objects
        .push(Annotation::new(0, Shape::Point(Point::new(0.0, 4.0))))
        .unwrap();
    let flipped = FlipHorizontal::default().apply(sample).unwrap().remove(0);
    assert_eq!(
        flipped.objects.get(0).unwrap().shape(),
        Shape::Point(Point::new(8.0, 4.0))
    );
    let mut objects = ObjectTable::default();
    for class_id in [0, 1, 0] {
        let mut a = Annotation::new(class_id, Shape::Rect(Rect::new(1.0, 1.0, 2.0, 2.0)));
        a.mask = Some(Mask::from_fn(8, 8, |x, y| x == class_id && y == 1).unwrap());
        objects.push(a).unwrap();
    }
    objects.retain(|o| o.class_id() == 1).unwrap();
    objects.validate().unwrap();
    assert_eq!(objects.len(), 1);
    assert!(objects.get(0).unwrap().mask().unwrap().get(1, 1));
}

#[test]
fn small_anisotropic_scales_do_not_preserve_circles() {
    let shape = Shape::Circle {
        center: Point::new(1000.0, 1000.0),
        radius: 100.0,
    };
    let affine = Affine {
        a: 0.0005,
        d: 0.001,
        ..Affine::IDENTITY
    };
    let output = affine.shape(&shape);
    assert!(matches!(output, Shape::Polygons(_)));
    assert!((output.bounds().width - 0.1).abs() < 1e-6);
    assert!((output.bounds().height - 0.2).abs() < 1e-6);
}
