---
id: gli-ky1e
status: closed
deps: []
links: []
created: 2026-04-10T01:21:12Z
type: feature
priority: 2
assignee: Paul English
parent: gli-jq1m
---
# Implement multi-device support for tch backend

Enable the tch backend to use multiple GPU devices by parsing device strings like cuda:0, cuda:1, etc. and distributing work across them.

## Design

1. Extend device parsing to support multiple devices
2. Create one TchExtractor per device
3. Round-robin or chunk-based distribution of work

