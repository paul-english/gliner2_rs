pub mod config;
pub mod decode;
pub mod engine;
pub mod extract;
pub mod preprocess;
pub mod processor;
pub mod schema;

#[cfg(feature = "candle")]
pub mod candle;

#[cfg(feature = "tch")]
pub mod tch_backend;

pub use config::ExtractorConfig;
pub use engine::Gliner2Engine;

pub use decode::Entity;
pub use extract::{
    BatchSchemaMode, ExtractOptions, batch_extract, extract_from_preprocessed, extract_with_schema,
};
#[cfg(feature = "candle")]
pub use candle::CandleExtractor;
pub use preprocess::{PreprocessedBatch, PreprocessedInput, TaskType, collate_preprocessed};
pub use processor::SchemaTransformer;
pub use schema::{
    ExtractionMetadata, ParsedFieldSpec, Schema, StructureBuilder, ValueDtype, create_schema,
    infer_metadata_from_schema, parse_field_spec,
};
#[cfg(feature = "tch")]
pub use tch_backend::TchExtractor;
