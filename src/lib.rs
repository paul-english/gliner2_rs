pub mod backends;
pub mod config;
pub mod decode;
pub mod engine;
pub mod extract;
pub mod preprocess;
pub mod processor;
pub mod schema;
pub mod setup;
pub mod span_utils;

pub use config::ExtractorConfig;
pub use engine::Gliner2Engine;

#[cfg(feature = "candle")]
pub use backends::candle::CandleExtractor;
#[cfg(feature = "tch")]
pub use backends::tch::TchExtractor;
#[cfg(feature = "tch")]
pub use backends::tch::parse_tch_device;
pub use decode::Entity;
pub use extract::{
    BatchSchemaMode, ExtractOptions, ExtractionOutput, LabelConfidence, TaskValue, batch_extract,
    batch_extract_streaming, extract_from_preprocessed, extract_with_schema,
};
pub use indexmap::IndexMap;
pub use preprocess::{PreprocessedBatch, PreprocessedInput, TaskType, collate_preprocessed};
pub use processor::SchemaTransformer;
pub use schema::{
    ClassificationLabelsInput, ClassificationTaskInfo, EntityTypeInfo, EntityTypesInput,
    ExtractionMetadata, ExtractionSchema, FieldSpecSource, ParsedFieldSpec, RegexMatchMode,
    RegexValidator, RelationTypeInfo, RelationTypesInput, Schema, SchemaDocument, SchemaInfo,
    StructureBuilder, StructureFieldInfo, StructureInfo, ValueDtype, create_schema,
    infer_metadata_from_schema, parse_field_spec,
};

// Compile-time assertions: backends and shared types must be Send+Sync for Rayon parallelism.
const _: () = {
    fn _assert_send_sync<T: Send + Sync>() {}

    #[cfg(feature = "candle")]
    fn _check_candle() {
        _assert_send_sync::<backends::candle::CandleExtractor>();
    }
    #[cfg(feature = "tch")]
    fn _check_tch() {
        _assert_send_sync::<backends::tch::TchExtractor>();
    }
    fn _check_transformer() {
        _assert_send_sync::<processor::SchemaTransformer>();
    }
};
