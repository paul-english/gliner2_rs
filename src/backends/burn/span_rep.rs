use super::layers::ProjectionLayer;
use super::weights::WeightMap;
use anyhow::Result;
use burn::prelude::*;
use burn::tensor::activation;

pub struct SpanMarkerV0<B: Backend> {
    project_start: ProjectionLayer<B>,
    project_end: ProjectionLayer<B>,
    out_project: ProjectionLayer<B>,
    max_width: usize,
}

impl<B: Backend> SpanMarkerV0<B> {
    pub fn load(
        map: &WeightMap,
        prefix: &str,
        _hidden_size: usize,
        max_width: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let project_start =
            ProjectionLayer::load(map, &format!("{prefix}.project_start"), device)?;
        let project_end = ProjectionLayer::load(map, &format!("{prefix}.project_end"), device)?;
        let out_project = ProjectionLayer::load(map, &format!("{prefix}.out_project"), device)?;
        Ok(Self {
            project_start,
            project_end,
            out_project,
            max_width,
        })
    }

    /// `h`: `[B, L, D]`, `span_idx`: `[B, S, 2]` int → `[B, L, max_width, D]`.
    pub fn forward(
        &self,
        h: &Tensor<B, 3>,
        span_idx: &Tensor<B, 3, Int>,
    ) -> Tensor<B, 4> {
        let [b, l, d] = h.dims();

        let start_rep = self.project_start.forward_3d(h); // [B, L, D]
        let end_rep = self.project_end.forward_3d(h); // [B, L, D]

        // span_idx: [B, S, 2] — extract start and end columns
        let starts: Tensor<B, 2, Int> = span_idx.clone().narrow(2, 0, 1).squeeze_dim::<2>(2); // [B, S]
        let ends: Tensor<B, 2, Int> = span_idx.clone().narrow(2, 1, 1).squeeze_dim::<2>(2); // [B, S]

        let start_span_rep = self.extract_elements(&start_rep, &starts); // [B, S, D]
        let end_span_rep = self.extract_elements(&end_rep, &ends); // [B, S, D]

        let cat = activation::relu(Tensor::cat(vec![start_span_rep, end_span_rep], 2));
        let out = self.out_project.forward_3d(&cat); // [B, S, D]

        out.reshape([b, l, self.max_width, d])
    }

    /// Gather elements from `h` `[B, L, D]` at indices `idx` `[B, S]` along dim 1.
    fn extract_elements(
        &self,
        h: &Tensor<B, 3>,
        idx: &Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let [b, _l, d] = h.dims();
        let s = idx.dims()[1];

        // Expand idx to [B, S, D] for gather on dim 1
        let expanded_idx: Tensor<B, 3, Int> = idx
            .clone()
            .unsqueeze_dim::<3>(2)
            .repeat_dim(2, d); // [B, S, D]
        h.clone().gather(1, expanded_idx)
    }
}
