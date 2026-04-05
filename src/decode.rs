use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub text: String,
    pub label: String,
    pub confidence: f32,
    pub start: usize, // char offset
    pub end: usize,   // char offset
}

/// Decodes span scores into entities. Greedy overlap suppression runs **per label** (same as
/// GliNER2 `_extract_entities`); results are concatenated in `labels` order.
pub fn find_spans(
    scores: &candle_core::Tensor,
    threshold: f32,
    labels: &[&str],
    text: &str,
    start_offsets: &[usize],
    end_offsets: &[usize],
) -> candle_core::Result<Vec<Entity>> {
    // scores: [NumEntities, L, max_width]
    let (num_entities, l, max_width) = scores.dims3()?;
    let scores_v = scores.to_vec3::<f32>()?;

    let mut out = Vec::new();

    for p in 0..num_entities {
        let label = labels[p];
        let mut per_label = Vec::new();
        for i in 0..l {
            for (j, &conf) in scores_v[p][i].iter().enumerate().take(max_width) {
                if conf >= threshold {
                    let end_token_idx = i + j;
                    if end_token_idx < l {
                        let char_start = start_offsets[i];
                        let char_end = end_offsets[end_token_idx];
                        let text_val = text[char_start..char_end].to_string();

                        per_label.push(Entity {
                            text: text_val,
                            label: label.to_string(),
                            confidence: conf,
                            start: char_start,
                            end: char_end,
                        });
                    }
                }
            }
        }
        out.extend(greedy_select(per_label));
    }

    Ok(out)
}

pub fn greedy_select(mut entities: Vec<Entity>) -> Vec<Entity> {
    // Sort by confidence descending
    entities.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut selected: Vec<Entity> = Vec::new();

    for entity in entities {
        let mut overlap = false;
        for s in &selected {
            // Overlap if not (entity.end <= s.start or entity.start >= s.end)
            if !(entity.end <= s.start || entity.start >= s.end) {
                overlap = true;
                break;
            }
        }
        if !overlap {
            selected.push(entity);
        }
    }

    selected
}

#[cfg(test)]
mod tests {
    use super::{Entity, greedy_select};

    #[test]
    fn per_label_greedy_keeps_overlapping_spans_for_different_labels() {
        let a = Entity {
            text: "foo".into(),
            label: "A".into(),
            confidence: 0.9,
            start: 0,
            end: 5,
        };
        let b = Entity {
            text: "bar".into(),
            label: "B".into(),
            confidence: 0.5,
            start: 2,
            end: 7,
        };

        let global = greedy_select(vec![a.clone(), b.clone()]);
        assert_eq!(global.len(), 1, "global NMS drops lower-confidence overlap");

        let mut per_label = greedy_select(vec![a]);
        per_label.extend(greedy_select(vec![b]));
        assert_eq!(per_label.len(), 2, "per-label NMS matches GliNER2 engine");
    }
}
