---
id: gli-4hsh
status: open
deps: []
links: []
created: 2026-04-05T16:15:02Z
type: task
priority: 2
assignee: Paul English
---
# Evaluate backends other than candle, a direct libtorch binding library is probably faster right now

Let's create a clean Adapter pattern around the existing candle code. Then let's make a new adapter that uses https://github.com/LaurentMazare/tch-rs

We'll want to update our throughput tests and reported output with this new backend. If tch-rs is faster we can default it as the backend implementation.
