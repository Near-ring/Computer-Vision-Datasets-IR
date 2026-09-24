//! Versioned, seekable packed IR: metadata + independently zstd-compressed MessagePack samples.
//! The UUID index is built by seeking over frames, without decoding any images.
use crate::{Category, Dataset, Metadata, Result, Sample, Uuid, invalid};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
};
pub mod wire;
const MAGIC: &[u8; 8] = b"CVDSIR02";
const LEGACY_MAGIC: &[u8; 8] = b"CVDSIR01";
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub compression_level: i32,
    pub max_record_bytes: u64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            compression_level: 3,
            max_record_bytes: 512 * 1024 * 1024,
        }
    }
}
#[derive(Serialize, Deserialize)]
struct Header {
    #[serde(default)]
    schema: String,
    #[serde(default)]
    version: u32,
    categories: Vec<Category>,
    metadata: Metadata,
    samples: u64,
}
fn encode<T: Serialize>(v: &T, options: Options) -> Result<Vec<u8>> {
    let raw = rmp_serde::to_vec_named(v).map_err(|e| invalid(e.to_string()))?;
    if raw.len() as u64 > options.max_record_bytes {
        return Err(invalid("packed record exceeds size limit"));
    }
    let mut encoder = zstd::stream::write::Encoder::new(Vec::new(), options.compression_level)?;
    encoder.include_checksum(true)?;
    encoder.write_all(&raw)?;
    let bytes = encoder.finish()?;
    if bytes.len() as u64 > options.max_record_bytes {
        return Err(invalid("compressed record exceeds size limit"));
    }
    Ok(bytes)
}
fn decode<T: DeserializeOwned>(bytes: &[u8], options: Options) -> Result<T> {
    let mut decoder = zstd::stream::read::Decoder::new(bytes)?;
    // Streaming encoders can declare a larger window than their small output.
    // Bound decoder workspace independently, with an 8 MiB minimum window.
    let window = options.max_record_bytes.max(8 * 1024 * 1024);
    let log = (64 - window.saturating_sub(1).leading_zeros()).min(31);
    decoder.window_log_max(log)?;
    let mut decoder = decoder.take(options.max_record_bytes.saturating_add(1));
    let mut raw = Vec::new();
    decoder.read_to_end(&mut raw)?;
    if raw.len() as u64 > options.max_record_bytes {
        return Err(invalid("decompressed record exceeds size limit"));
    }
    rmp_serde::from_slice(&raw).map_err(|e| invalid(format!("invalid packed record: {e}")))
}
fn read_u64(r: &mut impl Read) -> Result<u64> {
    let mut bytes = [0; 8];
    r.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}
fn read_frame(r: &mut impl Read, options: Options) -> Result<Vec<u8>> {
    let len = read_u64(r)?;
    if len > options.max_record_bytes || len > usize::MAX as u64 {
        return Err(invalid("packed frame too large"));
    }
    let mut bytes = Vec::new();
    r.take(len).read_to_end(&mut bytes)?;
    if bytes.len() as u64 != len {
        return Err(invalid("truncated packed frame"));
    }
    Ok(bytes)
}

pub fn write(dataset: &Dataset, path: impl AsRef<Path>, options: Options) -> Result<()> {
    dataset.validate()?;
    let file = File::create_new(path)?;
    let mut out = BufWriter::new(file);
    write_to(dataset, &mut out, options)?;
    out.flush()?;
    Ok(())
}
pub fn write_to(dataset: &Dataset, out: impl Write, options: Options) -> Result<()> {
    dataset.validate()?;
    let mut writer = StreamWriter::new(
        out,
        &dataset.categories,
        &dataset.metadata,
        dataset.samples.len() as u64,
        options,
    )?;
    for sample in &dataset.samples {
        writer.push(sample)?;
    }
    writer.finish()?;
    Ok(())
}

/// Bounded-memory writer for a known number of samples. Buffers one record at a time.
/// Call `finish` to check the count and flush. Dropping early leaves an incomplete file.
pub struct StreamWriter<W> {
    out: W,
    categories: Vec<Category>,
    expected: u64,
    written: u64,
    uids: HashSet<Uuid>,
    ids: HashSet<String>,
    options: Options,
}
impl<W: Write> StreamWriter<W> {
    pub fn new(
        mut out: W,
        categories: &[Category],
        metadata: &Metadata,
        samples: u64,
        options: Options,
    ) -> Result<Self> {
        Dataset {
            categories: categories.to_vec(),
            ..Default::default()
        }
        .validate()?;
        let header = encode(
            &Header {
                schema: "cv-dataset-ir".into(),
                version: 2,
                categories: categories.to_vec(),
                metadata: metadata.clone(),
                samples,
            },
            options,
        )?;
        out.write_all(MAGIC)?;
        out.write_all(&(header.len() as u64).to_le_bytes())?;
        out.write_all(&header)?;
        Ok(Self {
            out,
            categories: categories.to_vec(),
            expected: samples,
            written: 0,
            uids: HashSet::new(),
            ids: HashSet::new(),
            options,
        })
    }
    pub fn push(&mut self, sample: &Sample) -> Result<()> {
        sample.validate(&self.categories)?;
        if self.written >= self.expected
            || sample.uid.is_nil()
            || self.uids.contains(&sample.uid)
            || self.ids.contains(&sample.id)
        {
            return Err(invalid(
                "unexpected record count or duplicate sample identity",
            ));
        }
        let bytes = encode(&wire::Record::from_sample(sample), self.options)?;
        self.out.write_all(sample.uid.as_bytes())?;
        self.out.write_all(&(bytes.len() as u64).to_le_bytes())?;
        self.out.write_all(&bytes)?;
        self.uids.insert(sample.uid);
        self.ids.insert(sample.id.clone());
        self.written += 1;
        Ok(())
    }
    pub fn finish(mut self) -> Result<W> {
        if self.written != self.expected {
            return Err(invalid("packed sample count is incomplete"));
        }
        self.out.flush()?;
        Ok(self.out)
    }
}
pub fn read(path: impl AsRef<Path>) -> Result<Dataset> {
    Reader::open(path, Options::default())?.read_dataset()
}

pub struct Reader<R> {
    version: u32,
    source: R,
    header: Header,
    index: HashMap<Uuid, (u64, u64)>,
    order: Vec<Uuid>,
    options: Options,
}
impl Reader<BufReader<File>> {
    pub fn open(path: impl AsRef<Path>, options: Options) -> Result<Self> {
        Self::new(BufReader::new(File::open(path)?), options)
    }
}
impl<R: Read + Seek> Reader<R> {
    pub fn new(mut source: R, options: Options) -> Result<Self> {
        let mut magic = [0; 8];
        source.read_exact(&mut magic)?;
        if &magic != MAGIC && &magic != LEGACY_MAGIC {
            return Err(invalid("unknown packed IR magic/version"));
        }
        let header: Header = decode(&read_frame(&mut source, options)?, options)?;
        let version = if &magic == MAGIC { 2 } else { 1 };
        if version == 2 && (header.schema != "cv-dataset-ir" || header.version != 2) {
            return Err(invalid("packed schema mismatch"));
        }
        Dataset {
            categories: header.categories.clone(),
            ..Dataset::default()
        }
        .validate()?;
        let start = source.stream_position()?;
        let end = source.seek(SeekFrom::End(0))?;
        source.seek(SeekFrom::Start(start))?;
        if header.samples > (end - start) / 24 {
            return Err(invalid("packed sample count exceeds file length"));
        }
        let mut index = HashMap::new();
        let mut order = Vec::new();
        for _ in 0..header.samples {
            let mut bytes = [0; 16];
            source.read_exact(&mut bytes)?;
            let uid = Uuid::from_bytes(bytes);
            let len = read_u64(&mut source)?;
            let offset = source.stream_position()?;
            let next = offset
                .checked_add(len)
                .filter(|n| *n <= end)
                .ok_or_else(|| invalid("truncated packed sample"))?;
            if uid.is_nil()
                || len > options.max_record_bytes
                || len > usize::MAX as u64
                || index.insert(uid, (offset, len)).is_some()
            {
                return Err(invalid("duplicate UUID or invalid packed sample length"));
            }
            order.push(uid);
            source.seek(SeekFrom::Start(next))?;
        }
        if source.stream_position()? != end {
            return Err(invalid("trailing data in packed IR"));
        }
        Ok(Self {
            version,
            source,
            header,
            index,
            order,
            options,
        })
    }
    pub fn version(&self) -> u32 {
        self.version
    }
    pub fn categories(&self) -> &[Category] {
        &self.header.categories
    }
    pub fn metadata(&self) -> &Metadata {
        &self.header.metadata
    }
    pub fn sample_ids(&self) -> &[Uuid] {
        &self.order
    }
    pub fn read_sample(&mut self, uid: Uuid) -> Result<Option<Sample>> {
        let Some(&(offset, len)) = self.index.get(&uid) else {
            return Ok(None);
        };
        self.source.seek(SeekFrom::Start(offset))?;
        let mut bytes = vec![0; len as usize];
        self.source.read_exact(&mut bytes)?;
        let sample: Sample = if self.version == 1 {
            decode(&bytes, self.options)?
        } else {
            decode::<wire::Record>(&bytes, self.options)?.into_sample()?
        };
        if sample.uid != uid {
            return Err(invalid("packed UUID does not match sample"));
        }
        sample.validate(&self.header.categories)?;
        Ok(Some(sample))
    }
    pub fn read_dataset(mut self) -> Result<Dataset> {
        let mut samples = Vec::with_capacity(self.order.len());
        for uid in self.order.clone() {
            samples.push(self.read_sample(uid)?.expect("indexed UUID"));
        }
        let dataset = Dataset {
            categories: self.header.categories,
            metadata: self.header.metadata,
            samples,
        };
        dataset.validate()?;
        Ok(dataset)
    }
}

/// An uncompressed v2 MessagePack record, useful for language bindings.
pub fn encode_record(sample: &Sample) -> Result<Vec<u8>> {
    rmp_serde::to_vec_named(&wire::Record::from_sample(sample)).map_err(|e| invalid(e.to_string()))
}
pub fn decode_record(bytes: &[u8]) -> Result<Sample> {
    rmp_serde::from_slice::<wire::Record>(bytes)
        .map_err(|e| invalid(e.to_string()))?
        .into_sample()
}
