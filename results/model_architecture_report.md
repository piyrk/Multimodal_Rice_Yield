# Multimodal Deep-Learning Architecture Report

## Scope

This milestone implements and smoke-tests the architecture only. No model
training, hyperparameter tuning, or performance evaluation was performed.

The model consumes the 239 validated samples through the existing PyTorch
DataLoader:

- Satellite input: `[B, 7, 128, 128]`
- Weather input: `[B, T, 3]`, with `T=183` for the tested batch and a boolean
  padding mask
- Soil input: `[B, 4]`
- Output: `[B]`, one rice-yield prediction in kg/ha per sample

## Satellite CNN

`SatelliteCNN` uses four compact convolution blocks:

1. `7 -> 32` channels
2. `32 -> 64` channels
3. `64 -> 128` channels
4. `128 -> 192` channels

Each block contains two 3x3 convolutions, BatchNorm, ReLU, and 2x2 max
pooling. Adaptive global average pooling reduces the final spatial map to one
value per channel, avoiding a large flatten-to-linear layer. A linear
projection, LayerNorm, and ReLU produce a **128-dimensional satellite
embedding**.

Output: `[B, 128]`.

## Weather LSTM

`WeatherLSTM` processes the three daily variables:

- rainfall_mm
- temperature_c
- humidity_pct

It uses a one-layer LSTM with hidden size 64. The existing
`weather_mask` is converted to sequence lengths and passed to
`pack_padded_sequence`, so padded daily values do not contribute to the
recurrent computation. A projection layer produces a **128-dimensional
weather embedding**.

Output: `[B, 128]`.

## Soil MLP

`SoilMLP` uses:

- Linear `4 -> 32`
- LayerNorm and ReLU
- Linear `32 -> 64`
- LayerNorm and ReLU

Output: `[B, 64]`.

## Attention-Based Fusion

Each branch is projected into a common **128-dimensional fusion space**:

- Satellite: `128 -> 128`
- Weather: `128 -> 128`
- Soil: `64 -> 128`

The three projected embeddings are stacked as `[B, 3, 128]`. A shared
learnable scoring network produces one scalar score per modality. Softmax
normalizes the three scores into sample-specific attention weights. The fused
representation is the weighted sum:

`fused = sum(attention_weight_m * modality_embedding_m)`.

The weights are learned from the data; no manual modality weighting is used.
This mechanism is straightforward to describe as learned modality attention
in a research paper.

Output: fused representation `[B, 128]`, attention weights `[B, 3]`.

## Regression Head

The regression head contains:

- LayerNorm `128`
- Linear `128 -> 64`
- ReLU
- Dropout `0.1`
- Linear `64 -> 1`

The final singleton dimension is removed, producing `[B]`. The value is a
continuous rice-yield prediction in kg/ha.

## Parameter and Memory Summary

- Trainable parameters: **954,242**
- Approximate parameter storage: **3.640 MB** using float32 parameters
- Batch size used in the forward smoke test: 8
- No optimizer state or activation-memory estimate is included in the
  parameter-size figure.

The design is intentionally compact for an RTX 3050 4 GB GPU: global average
pooling prevents a large fully connected satellite head, the LSTM has one
layer and 64 hidden units, and the soil branch is small. Batch size, mixed
precision, and gradient accumulation can be selected later during training
based on measured memory usage.

## Forward-Test Result

`src/models/test_model_forward.py` loaded three DataLoader batches and ran all
three through the model on CPU:

```text
satellite embedding shape = (8, 128)
weather embedding shape = (8, 128)
soil embedding shape = (8, 64)
fused representation shape = (8, 128)
attention weights shape = (8, 3)
prediction shape = (8,)
prediction finite = True
trainable parameters = 954242
approximate parameter size MB = 3.640
CPU forward pass = PASSED
multiple batches loaded = PASSED
```

CUDA was checked and skipped because CUDA was unavailable in the current
environment. No model weights were trained or downloaded.

## Limitations

- Weather normalization has not been applied; that belongs in a later
  training-only preprocessing stage.
- The current attention fusion is a weighted sum rather than a transformer
  cross-modal interaction block, chosen for interpretability and memory
  efficiency.
- The model is untrained, so these forward-pass checks do not establish
  predictive quality or performance metrics.
