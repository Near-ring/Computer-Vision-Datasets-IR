//! A small deterministic fixture for Rust/Python interoperability checks.
use cv_dataset_ir::*;
fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("output path");
    let mut category = Category::new("plant");
    category.keypoints = vec!["tip".into(), "root".into()];
    category.skeleton = vec![[0, 1]];
    let mut dataset = Dataset {
        categories: vec![category],
        ..Default::default()
    };
    dataset.metadata.insert(
        "nested".into(),
        serde_json::json!({"version":1,"tags":["a",true,null]}),
    );
    for i in 0..3 {
        let image = Raster::new(8, 6, (0..8 * 6 * 3).map(|v| v as u8).collect())?;
        let mut sample = Sample::new(format!("frame-{i}"), image);
        sample.uid = Uuid::from_u128(i + 1);
        sample.split = [Split::Train, Split::Val, Split::Named("train".into())][i as usize].clone();
        sample
            .metadata
            .insert("operation".into(), "user value, preserve me".into());
        let mut a = Annotation::new(
            0,
            Shape::Circle {
                center: Point::new(3.0, 3.0),
                radius: 2.5,
            },
        );
        a.id = 19;
        a.mask = Some(Mask::from_fn(8, 6, |x, y| {
            (2..6).contains(&x) && (1..5).contains(&y) && !(x == 3 && y == 2)
        })?);
        a.keypoints = vec![
            Keypoint::new(0.0, 0.0, Visibility::Occluded),
            Keypoint::new(4.0, 3.0, Visibility::Visible),
        ];
        a.metadata.insert(
            "quality".into(),
            serde_json::json!({"score":0.9,"reviewed":true}),
        );
        sample.objects.push(a)?;
        sample.objects.push(Annotation::new(
            0,
            Shape::Polygons(vec![vec![
                Point::new(1.0, 1.0),
                Point::new(2.0, 1.0),
                Point::new(1.0, 2.0),
            ]]),
        ))?;
        dataset.samples.push(sample);
    }
    packed::write(&dataset, path, Default::default())
}
