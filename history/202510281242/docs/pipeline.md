# Pipeline

The pipeline uses small modules under src/ to keep IO, tracking,
counting, and metrics separate.

## Stages
- io: resolve stream or local video
- tracking: wrap YOLO track results
- counting: count unique IDs or line crossings
- metrics: fps and simple counters

Default count mode: line
