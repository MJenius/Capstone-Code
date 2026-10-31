# ANN Training Report

Generated: 2026-10-31T15:01:56

| Item | Value |
|---|---|
| Device | cpu (torch 2.13.0+cpu) |
| Training samples | 2240 |
| Validation samples | 360 |
| Epochs requested | 5 |
| Epochs completed (total, incl. resumed) | 5 |
| Early stopped | False |
| Quick mode (40 batches/epoch) | False |
| Batch size / LR | 32 / 0.002 |
| Loss | BCEWithLogits + 0.1 * (1 - softNC) |
| Optimizer / scheduler | AdamW (wd 1e-4) / ReduceLROnPlateau(mode=max on val NC, factor=0.5, patience=2) |
| Selection criterion | max validation NC (test split untouched) |
| Seed | 42 |
| Wall time this run | 968 s |
| Parameters | 276,466 |

## Best validation epoch (4)

| Metric | Value |
|---|---|
| loss | 0.3034 |
| nc | 0.7333 |
| ber | 0.1299 |
| accuracy | 0.8701 |
| precision | 0.8428 |
| recall | 0.8313 |
| f1 | 0.8281 |

## Per-epoch history

| epoch | train loss | val loss | val NC | val BER | val acc | val F1 | lr | s |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.6004 | 0.3871 | 0.6813 | 0.1652 | 0.8348 | 0.8121 | 3.00e-03 | 352.1 |
| 2 | 0.3561 | 0.3307 | 0.7169 | 0.1384 | 0.8616 | 0.8248 | 3.00e-03 | 350.7 |
| 3 | 0.3339 | 0.3138 | 0.7305 | 0.1351 | 0.8649 | 0.8401 | 3.00e-03 | 324.4 |
| 4 | 0.3173 | 0.3034 | 0.7333 | 0.1299 | 0.8701 | 0.8281 | 3.00e-03 | 322.2 |
| 5 | 0.3077 | 0.3063 | 0.7272 | 0.1323 | 0.8677 | 0.8013 | 3.00e-03 | 320.7 |

Note: validation metrics are measured on held-out *validation hosts* and watermark IDs.
They are used for model selection only; final claims use the isolated test split (ann-eval / comparison).
