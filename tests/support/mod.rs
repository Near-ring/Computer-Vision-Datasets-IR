#![allow(dead_code)] // Shared fixtures are used by different integration-test binaries.
use cv_dataset_ir::*;
pub fn fixture() -> Dataset {
    let mut category = Category::new("plant");
    category.keypoints = vec!["left".into(), "right".into()];
    category.skeleton = vec![[0, 1]];
    let mut dataset = Dataset {
        categories: vec![category, Category::new("weed")],
        ..Dataset::default()
    };
    for (i, split) in [Split::Train, Split::Val, Split::Test]
        .into_iter()
        .enumerate()
    {
        let image =
            Raster::new(16, 12, (0..16 * 12 * 3).map(|x| (x % 251) as u8).collect()).unwrap();
        let mut sample = Sample::new(format!("frame-{i}"), image);
        sample.split = split;
        sample.metadata.insert("camera".into(), "rgb-1".into());
        let mut object = Annotation::new(0, Shape::Rect(Rect::new(2.0, 2.0, 8.0, 6.0)));
        object.id = 42;
        object.keypoints = vec![
            Keypoint::new(3.0, 3.0, Visibility::Visible),
            Keypoint::new(9.0, 6.0, Visibility::Occluded),
        ];
        sample.objects.push(object).unwrap();
        let mut object = Annotation::new(
            1,
            Shape::Polygons(vec![vec![
                Point::new(10.0, 3.0),
                Point::new(14.0, 3.0),
                Point::new(13.0, 9.0),
            ]]),
        );
        object.id = 43;
        sample.objects.push(object).unwrap();
        dataset.samples.push(sample);
    }
    dataset.metadata.insert("purpose".into(), "fixture".into());
    dataset.validate().unwrap();
    dataset
}
pub fn detection() -> Dataset {
    let mut d = fixture();
    for s in &mut d.samples {
        let mut objects = ObjectTable::default();
        for o in s.objects.iter() {
            let mut a = Annotation::new(o.class_id(), Shape::Rect(o.bbox()));
            a.id = o.id();
            objects.push(a).unwrap();
        }
        s.objects = objects;
    }
    for c in &mut d.categories {
        c.keypoints.clear();
        c.skeleton.clear();
    }
    d
}
pub fn pose() -> Dataset {
    let mut d = fixture();
    d.categories.truncate(1);
    for s in &mut d.samples {
        s.objects.retain(|o| o.class_id() == 0).unwrap();
    }
    d
}
pub fn segmentation() -> Dataset {
    let mut d = fixture();
    for s in &mut d.samples {
        let mut t = ObjectTable::default();
        for o in s.objects.iter() {
            let mut a = Annotation::new(o.class_id(), Shape::Polygons(o.shape().polygons(64)));
            a.id = o.id();
            t.push(a).unwrap();
        }
        s.objects = t;
    }
    for c in &mut d.categories {
        c.keypoints.clear();
        c.skeleton.clear();
    }
    d
}
pub fn assert_identity(a: &Dataset, b: &Dataset) {
    assert_eq!(a.samples.len(), b.samples.len());
    let index = b.index().unwrap();
    for s in &a.samples {
        let t = index.get(s.uid).unwrap();
        assert_eq!(s.id, t.id);
        assert_eq!(s.split, t.split);
        assert_eq!(s.image, t.image);
        assert_eq!(s.objects.len(), t.objects.len());
    }
}
