use anyhow::Result;
use regex::Regex;
use tokenizers::Tokenizer;
use std::collections::HashSet;

pub struct WhitespaceTokenSplitter {
    re: Regex,
}

impl WhitespaceTokenSplitter {
    pub fn new() -> Result<Self> {
        let re = Regex::new(r"(?xi)
            (?:https?://[^\s]+|www\.[^\s]+)
            |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
            |@[a-z0-9_]+
            |\w+(?:[-_]\w+)*
            |\S
        ")?;
        Ok(Self { re })
    }

    pub fn split<'a>(&self, text: &'a str) -> Vec<(&'a str, usize, usize)> {
        self.re.find_iter(text)
            .map(|m| (m.as_str(), m.start(), m.end()))
            .collect()
    }
}

pub struct SchemaTransformer {
    pub tokenizer: Tokenizer,
    pub(crate) word_splitter: WhitespaceTokenSplitter,
    pub(crate) special_token_ids: HashSet<u32>,
}

pub const SEP_TEXT: &str = "[SEP_TEXT]";
pub const P_TOKEN: &str = "[P]";
pub const E_TOKEN: &str = "[E]";

impl SchemaTransformer {
    pub fn new(tokenizer_path: &str) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        
        // Add special tokens if they are not already there
        // In practice, for from_file, they should be in the tokenizer.json already
        // But for GLiNER2, we might need to ensure they exist.
        
        let word_splitter = WhitespaceTokenSplitter::new()?;
        
        let mut special_token_ids = HashSet::new();
        for tok in &[P_TOKEN, E_TOKEN, "[C]", "[R]", "[L]"] {
            if let Some(id) = tokenizer.token_to_id(tok) {
                special_token_ids.insert(id);
            }
        }

        Ok(Self {
            tokenizer,
            word_splitter,
            special_token_ids,
        })
    }

    pub fn format_input_for_ner(&self, text: &str, entities: &[&str]) -> Result<FormattedInput> {
        // Build schema: [P] entities ( [E] person [E] location ) [SEP_TEXT]
        let mut schema_tokens = vec![P_TOKEN.to_string(), "entities".to_string(), "(".to_string()];
        for entity in entities {
            schema_tokens.push(E_TOKEN.to_string());
            schema_tokens.push(entity.to_string());
        }
        schema_tokens.push(")".to_string());
        schema_tokens.push(SEP_TEXT.to_string());

        let mut subwords = Vec::new();
        let mut input_ids = Vec::new();
        let mut text_word_first_positions = Vec::new();
        let mut schema_special_positions = Vec::new();
        
        // Process schema
        for token in &schema_tokens {
            let encoded = self.tokenizer.encode(token.as_str(), false)
                .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
            
            let ids = encoded.get_ids();
            let tokens = encoded.get_tokens();
            
            for (&id, subword) in ids.iter().zip(tokens.iter()) {
                let pos = input_ids.len();
                input_ids.push(id);
                subwords.push(subword.clone());
                
                if self.special_token_ids.contains(&id) {
                    schema_special_positions.push(pos);
                }
            }
        }

        let text_start_idx = input_ids.len();

        // Process text
        let words = self.word_splitter.split(text);
        let mut start_offsets = Vec::new();
        let mut end_offsets = Vec::new();

        for (word, start, end) in words {
            let encoded = self.tokenizer.encode(word.to_lowercase().as_str(), false)
                .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
            
            let ids = encoded.get_ids();
            let tokens = encoded.get_tokens();
            
            if !ids.is_empty() {
                text_word_first_positions.push(input_ids.len());
                for (&id, subword) in ids.iter().zip(tokens.iter()) {
                    input_ids.push(id);
                    subwords.push(subword.clone());
                }
                start_offsets.push(start);
                end_offsets.push(end);
            }
        }

        Ok(FormattedInput {
            input_ids,
            text_word_first_positions,
            schema_special_positions,
            start_offsets,
            end_offsets,
            text_start_idx,
        })
    }
}

pub struct FormattedInput {
    pub input_ids: Vec<u32>,
    pub text_word_first_positions: Vec<usize>,
    pub schema_special_positions: Vec<usize>,
    pub start_offsets: Vec<usize>,
    pub end_offsets: Vec<usize>,
    pub text_start_idx: usize,
}
