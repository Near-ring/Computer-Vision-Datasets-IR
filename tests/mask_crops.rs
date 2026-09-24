use cv_dataset_ir::{
    packed::{self, wire},
    transforms::{MaskCrops, Pass, Resize},
    *,
};
use std::io::Cursor;

fn fixture() -> Dataset {
    packed::Reader::new(
        Cursor::new(include_bytes!("fixtures/legacy_v1.cvir")),
        Default::default(),
    )
    .unwrap()
    .read_dataset()
    .unwrap()
}

#[test]
fn tight_mask_crop_keeps_pixels_holes_pose_geometry_and_metadata() {
    let dataset = fixture();
    let source = &dataset.samples[0];
    let original = source.objects.get(0).unwrap();
    let fill = [201, 202, 203];
    let crop = MaskCrops {
        fill,
        ..Default::default()
    }
    .crop_instance(source, 0)
    .unwrap();
    assert_eq!((crop.image.width(), crop.image.height()), (4, 4));
    assert_eq!(crop.objects.len(), 1);
    assert_eq!(crop.metadata, source.metadata);
    assert_eq!(crop.split, source.split);
    assert_ne!(crop.uid, source.uid);
    let object = crop.objects.get(0).unwrap();
    assert_eq!(object.id(), original.id());
    assert_eq!(object.class_id(), original.class_id());
    assert_eq!(object.metadata(), original.metadata());
    assert_eq!(object.is_crowd(), original.is_crowd());
    assert_eq!(
        object.shape(),
        Shape::Circle {
            center: Point::new(1., 2.),
            radius: 2.5
        }
    );
    assert_eq!(
        object.keypoints()[0],
        Keypoint::new(-2., -1., Visibility::Occluded)
    );
    assert_eq!(
        object.keypoints()[1],
        Keypoint::new(2., 2., Visibility::Visible)
    );
    assert_eq!(object.mask().unwrap().area(), 15);
    assert!(!object.mask().unwrap().get(1, 1));
    for y in 0..4u32 {
        for x in 0..4u32 {
            let actual =
                &crop.image.pixels()[((y * 4 + x) * 3) as usize..((y * 4 + x) * 3 + 3) as usize];
            if (x, y) == (1, 1) {
                assert_eq!(actual, fill);
            } else {
                let i = (((y + 1) * 8 + x + 2) * 3) as usize;
                assert_eq!(actual, &source.image.pixels()[i..i + 3]);
            }
        }
    }
    assert_eq!(crop.provenance[0].parent_uid, source.uid);
    assert_eq!(crop.provenance[0].operation, "mask_crop");
    assert_eq!(
        crop.provenance[0].parameters["origin"],
        serde_json::json!([2, 1])
    );
    assert_eq!(crop.provenance[0].parameters["object_id"], 19);
    crop.validate(&dataset.categories).unwrap();
    let decoded = packed::decode_record(&packed::encode_record(&crop).unwrap()).unwrap();
    assert_eq!(crop, decoded);
}

#[test]
fn padding_external_masks_empty_masks_and_class_selection() {
    let d = fixture();
    let s = &d.samples[2];
    let padded = MaskCrops {
        padding: u32::MAX,
        fill: [7; 3],
        ..Default::default()
    }
    .crop_instance(s, 0)
    .unwrap();
    assert_eq!((padded.image.width(), padded.image.height()), (8, 6));
    assert_eq!(&padded.image.pixels()[..3], &[7; 3]);
    assert_eq!(padded.split, Split::Named("train".into()));
    assert_eq!(
        padded.objects.get(0).unwrap().keypoints(),
        s.objects.get(0).unwrap().keypoints()
    );
    let pass = MaskCrops::default();
    assert_eq!(pass.apply(s.clone()).unwrap().len(), 1);
    assert!(
        MaskCrops {
            classes: Some(vec![99]),
            ..Default::default()
        }
        .apply(s.clone())
        .unwrap()
        .is_empty()
    );
    let mask = Mask::from_fn(8, 6, |x, y| x == 7 && y == 5).unwrap();
    let crop = pass.crop(s, 1, &mask).unwrap();
    assert_eq!(crop.image.pixels(), &s.image.pixels()[141..]);
    assert_eq!(
        crop.objects.get(0).unwrap().metadata(),
        s.objects.get(1).unwrap().metadata()
    );
    assert!(pass.crop_instance(s, 1).is_err());
    assert!(pass.crop(s, 99, &mask).is_err());
    assert!(
        pass.crop(s, 0, &Mask::from_fn(1, 1, |_, _| false).unwrap())
            .is_err()
    );
    let empty = Mask::from_fn(8, 6, |_, _| false).unwrap();
    assert!(pass.crop(s, 0, &empty).is_err());
    let mut sample = s.clone();
    let mut a = sample.objects.get(0).unwrap().to_owned();
    a.mask = Some(empty);
    sample.objects = ObjectTable::default();
    sample.objects.push(a).unwrap();
    assert!(pass.apply(sample).unwrap().is_empty());
}

#[test]
fn transform_history_preserves_user_keys_and_survives_v2() {
    let mut d = fixture();
    let parent = d.samples[0].uid;
    let crop = MaskCrops::default()
        .crop_instance(&d.samples[0], 0)
        .unwrap();
    let crop_uid = crop.uid;
    let resized = Resize {
        width: 8,
        height: 8,
    }
    .apply(crop)
    .unwrap()
    .remove(0);
    assert_eq!(resized.metadata["operation"], "user value, preserve me");
    assert_eq!(resized.provenance.len(), 2);
    assert_eq!(resized.provenance[0].parent_uid, parent);
    assert_eq!(resized.provenance[1].parent_uid, crop_uid);
    d.samples = vec![resized];
    let mut bytes = Vec::new();
    packed::write_to(&d, &mut bytes, Default::default()).unwrap();
    assert_eq!(&bytes[..8], b"CVDSIR02");
    let reader = packed::Reader::new(Cursor::new(bytes), Default::default()).unwrap();
    assert_eq!(reader.version(), 2);
    assert_eq!(reader.read_dataset().unwrap(), d);
}

#[test]
fn legacy_file_migrates_and_wire_rejects_invalid_columns() {
    let d = fixture();
    assert_eq!(d.samples.len(), 3);
    assert!(d.samples[0].provenance.is_empty());
    let s = &d.samples[0];
    let wire = wire::Record::from_sample(s);
    assert_eq!(wire.image.dtype, "|u1");
    assert_eq!(wire.image.shape, [6, 8, 3]);
    assert_eq!(wire.objects.ids.uint64(), [19, 0]);
    assert_eq!(wire.objects.polygon_object_offsets.uint64(), [0, 0, 1]);
    assert_eq!(wire.objects.keypoint_offsets.uint64(), [0, 2, 2]);
    assert_eq!(wire.clone().into_sample().unwrap(), *s);
    let mut bad = wire.clone();
    bad.objects.boxes.dtype = ">f4".into();
    assert!(bad.into_sample().is_err());
    let mut bad = wire.clone();
    bad.objects.keypoint_offsets.data[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(bad.into_sample().is_err());
    let mut bad = wire.clone();
    bad.objects.shape_params.data[12..16].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(bad.into_sample().is_err());
    let mut bad = wire.clone();
    bad.objects.visibility.data[0] = 3;
    assert!(bad.into_sample().is_err());
    let mut bad = wire.clone();
    bad.objects.masks[0].as_mut().unwrap().bitorder = "big".into();
    assert!(bad.into_sample().is_err());
    let mut bad = wire.clone();
    bad.image.shape[0] = usize::MAX;
    assert!(bad.into_sample().is_err());
    let mut bad = wire;
    bad.split.kind = "invalid".into();
    assert!(bad.into_sample().is_err());
}
