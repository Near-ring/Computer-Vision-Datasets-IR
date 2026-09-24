//! Minimal library usage. `cargo run --example pipeline` needs no external data.
use cv_dataset_ir::{
    packed,
    transforms::{Filter, InstanceCrops, Pipeline, Resize},
    *,
};
fn main() -> Result<()> {
    let mut sample = Sample::new(
        "camera/frame-001",
        Raster::new(640, 480, vec![128; 640 * 480 * 3])?,
    );
    sample.split = Split::Train;
    sample.objects.push(Annotation::new(
        0,
        Shape::Rect(Rect::new(100.0, 120.0, 160.0, 180.0)),
    ))?;
    let source_uid = sample.uid;
    let dataset = Dataset {
        categories: vec![Category::new("plant")],
        samples: vec![sample],
        ..Default::default()
    };
    assert!(dataset.index()?.get(source_uid).is_some());
    let pipeline = Pipeline::new()
        .then(Filter {
            min_area: 100.0,
            ..Default::default()
        })
        .then(InstanceCrops {
            padding: 0.1,
            keep_neighbors: false,
            ..Default::default()
        })
        .then(Resize {
            width: 224,
            height: 224,
        });
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("thread pool");
    let output = pool.install(|| pipeline.run(dataset))?;
    let mut buffer = vec![];
    packed::write_to(&output, &mut buffer, Default::default())?;
    let mut reader = packed::Reader::new(std::io::Cursor::new(buffer), Default::default())?;
    let found = reader
        .read_sample(output.samples[0].uid)?
        .expect("saved sample");
    println!(
        "source={source_uid}; crop={}; size={}x{}; split={:?}",
        found.uid,
        found.image.width(),
        found.image.height(),
        found.split
    );
    Ok(())
}
