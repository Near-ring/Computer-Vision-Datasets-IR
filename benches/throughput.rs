use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use cv_dataset_ir::{
    transforms::{Filter, Pipeline, Resize},
    *,
};
use std::hint::black_box;
fn dataset() -> Dataset {
    let mut d = Dataset {
        categories: vec![Category::new("object")],
        ..Default::default()
    };
    for i in 0..32 {
        let pixels = (0..640 * 480 * 3).map(|i| (i % 251) as u8).collect();
        let mut s = Sample::new(i.to_string(), Raster::new(640, 480, pixels).unwrap());
        for j in 0..100 {
            s.objects
                .push(Annotation::new(
                    0,
                    Shape::Rect(Rect::new(
                        (j % 10 * 50) as f32,
                        (j / 10 * 40) as f32,
                        30.0,
                        20.0,
                    )),
                ))
                .unwrap();
        }
        d.samples.push(s);
    }
    d
}
fn benchmarks(c: &mut Criterion) {
    let d = dataset();
    let mut group = c.benchmark_group("dataset");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(2));
    group.throughput(Throughput::Elements(32));
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let pipeline = Pipeline::new().then(Resize {
            width: 320,
            height: 240,
        });
        group.bench_function(format!("resize_{threads}_threads"), |b| {
            b.iter_batched(
                || d.clone(),
                |d| black_box(pool.install(|| pipeline.run(d).unwrap())),
                BatchSize::LargeInput,
            )
        });
    }
    group.bench_function("filter_3200_boxes", |b| {
        b.iter_batched(
            || d.clone(),
            |d| {
                black_box(
                    Pipeline::new()
                        .then(Filter {
                            min_area: 1000.0,
                            ..Default::default()
                        })
                        .run(d)
                        .unwrap(),
                )
            },
            BatchSize::LargeInput,
        )
    });
    group.finish();
    let index = d.index().unwrap();
    let uid = d.samples[16].uid;
    c.bench_function("uuid_lookup", |b| {
        b.iter(|| black_box(index.get(black_box(uid))))
    });
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);
