# Third-Party Notices

## MediaPipe canonical face model

`core/analytics.py` includes a centered, unit-scaled two-dimensional projection
of 68 vertices selected from MediaPipe's canonical face model:

- Source: `mediapipe/modules/face_geometry/data/canonical_face_model.obj`
- Source commit: `a908d668c730da128dfa8d9f6bd25d519d006692`
- Copyright: Copyright 2020 The MediaPipe Authors
- License: Apache License 2.0
- Project: https://github.com/google-ai-edge/mediapipe

The selected coordinates were transformed by retaining the horizontal and
vertical components, reversing the vertical axis for image coordinates,
centering the 68 points, and scaling them to unit centroid size. The resulting
template remains a synthetic geometric seed; it is not a person, population
estimate, biological norm, or appearance standard.

The full Apache License 2.0 text is provided in
`LICENSES/Apache-2.0.txt`. MediaPipe and Google do not endorse this project.
