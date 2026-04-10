---
id: gli-2pwt
status: closed
deps: []
links: []
created: 2026-04-10T01:21:08Z
type: feature
priority: 2
assignee: Paul English
parent: gli-jq1m
---
# Add --num-workers CLI flag for parallel execution

Add a new CLI flag --num-workers (default 1) that enables parallel execution across multiple CPU cores or GPU devices. When > 1, the system will create multiple engine instances and distribute work across them.

## Design

1. Add --num-workers usize flag to Cli struct
2. When num_workers > 1, create multiple engine instances
3. Distribute text batches across workers using rayon
4. Each worker processes its batch independently and results are merged

