---
id: gli-7so4
status: closed
deps: []
links: []
created: 2026-04-10T01:21:16Z
type: feature
priority: 3
assignee: Paul English
parent: gli-jq1m
---
# Optimize parquet I/O for parallel processing

When processing parquet files with multiple workers, enable parallel reading and writing of parquet data to maximize throughput.

## Design

1. Use parquet's built-in parallelism for reading
2. Stream results to parquet output in parallel batches
3. Consider file partitioning strategies for large datasets

