use candle_core::{Result, Tensor, D};
use candle_nn::{Activation, Module, Sequential, VarBuilder};
use crate::layers::create_projection_layer;

pub struct SpanMarkerV0 {
    project_start: Sequential,
    project_end: Sequential,
    out_project: Sequential,
    max_width: usize,
    hidden_size: usize,
}

impl SpanMarkerV0 {
    pub fn load(hidden_size: usize, max_width: usize, vb: VarBuilder) -> Result<Self> {
        let project_start = create_projection_layer(hidden_size, hidden_size, vb.pp("project_start"))?;
        let project_end = create_projection_layer(hidden_size, hidden_size, vb.pp("project_end"))?;
        let out_project =
            create_projection_layer(hidden_size * 2, hidden_size, vb.pp("out_project"))?;

        Ok(Self {
            project_start,
            project_end,
            out_project,
            max_width,
            hidden_size,
        })
    }

    pub fn forward(&self, h: &Tensor, span_idx: &Tensor) -> Result<Tensor> {
        // h: [B, L, D]
        // span_idx: [B, S, 2] where S = L * max_width
        let (b, l, d) = h.dims3()?;
        
        let start_rep = self.project_start.forward(h)?; // [B, L, D]
        let end_rep = self.project_end.forward(h)?;     // [B, L, D]

        let starts = span_idx.get_on_dim(D::Minus1, 0)?; // [B, S]
        let ends = span_idx.get_on_dim(D::Minus1, 1)?;   // [B, S]

        let start_span_rep = self.extract_elements(&start_rep, &starts)?; // [B, S, D]
        let end_span_rep = self.extract_elements(&end_rep, &ends)?;     // [B, S, D]

        let cat = Tensor::cat(&[&start_span_rep, &end_span_rep], D::Minus1)?.apply(&Activation::Relu)?;
        
        let out = self.out_project.forward(&cat)?; // [B, S, D]
        
        out.reshape((b, l, self.max_width, d))
    }

    fn extract_elements(&self, h: &Tensor, idx: &Tensor) -> Result<Tensor> {
        // h: [B, L, D]
        // idx: [B, S]
        // result: [B, S, D]
        
        let (b, l, d) = h.dims3()?;
        let s = idx.dim(1)?;
        
        // We need to gather across L dimension for each B and D.
        // Candle's gather is a bit different from PyTorch's.
        // Tensor::gather(self, indexes, dim)
        
        // We want to pick elements from dim 1 (L) using idx.
        // Since idx is [B, S], we need to broadcast/expand it to match [B, S, D]?
        // No, gather works by taking values from `self` at `indexes` along `dim`.
        // If dim=1, it takes h[b, idx[b, s, d], d]
        
        let expanded_idx = idx.unsqueeze(2)?.expand(&[b, s, d])?.contiguous()?;
        h.contiguous()?.gather(&expanded_idx, 1)
    }
}
