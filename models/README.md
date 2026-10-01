# Vendored models

## `basic_pitch_nmp.onnx`

Basic Pitch (ICASSP 2022) note/onset/contour model by Spotify AB.

- Source: `basic-pitch==0.4.0` wheel, `basic_pitch/saved_models/icassp_2022/nmp.onnx`
- License: Apache-2.0 (see `LICENSE.basic-pitch`, `NOTICE.basic-pitch`)
- SHA-256: `2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec`
- Input: `serving_default_input_2:0`, float32 `(batch, 43844, 1)` at 22 050 Hz
- Outputs: `StatefulPartitionedCall:1` = note `(batch, T, 88)`,
  `:2` = onset `(batch, T, 88)`, `:0` = contour `(batch, T, 264)`

The app runs this model directly through ONNX Runtime (CUDA or CPU). The
`basic-pitch` Python package is used only in the dev lab as a reference oracle
to generate golden outputs (`scripts/gen_golden.py`).
