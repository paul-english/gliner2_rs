pub mod backends;
pub mod config;
pub mod decode;
pub mod engine;
pub mod extract;
pub mod preprocess;
pub mod processor;
pub mod schema;

pub use config::ExtractorConfig;
pub use engine::Gliner2Engine;

#[cfg(feature = "candle")]
pub use backends::candle::CandleExtractor;
#[cfg(feature = "tch")]
pub use backends::tch::TchExtractor;
pub use decode::Entity;
pub use extract::{
    BatchSchemaMode, ExtractOptions, batch_extract, extract_from_preprocessed, extract_with_schema,
};
pub use preprocess::{PreprocessedBatch, PreprocessedInput, TaskType, collate_preprocessed};
pub use processor::SchemaTransformer;
pub use schema::{
    ExtractionMetadata, ParsedFieldSpec, RegexMatchMode, RegexValidator, Schema, StructureBuilder,
    ValueDtype, create_schema, infer_metadata_from_schema, parse_field_spec,
};
