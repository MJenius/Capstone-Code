# ANN Dataset Generation & Training Methodology

## 1. Resolution of Legacy Dataset Inconsistencies

During the technical audit (`docs/technical_audit.md`), three critical flaws were identified in the legacy Phase-2 dataset generator (`create_training_data.py` and `phase_runner.py` Phase 8):

1. **Target Shape & Semantics Mismatch:** The legacy script saved $256 \times 256$ host-subtracted clean residuals `(watermarked - host) / alpha_base + 0.5` as labels while `core/ann_interface.py` specified a $32 \times 32$ binary watermark output.
2. **Host-Image Split Leakage:** The $1,100$ files in `preprocessed/I_channel/` come from $800$ unique DIV2K source images (`0001`–`0800`), with $300$ duplicate counter files. Splitting by the $1,100$ file names leaked **59 identical host images between train and test** and **53 between train and validation**.
3. **Single-Watermark Memorization:** Every legacy sample embedded the same canonical watermark (`PW26_PAC_01`). Any neural network trained on constant labels learns to output that constant pattern regardless of the input image.

---

## 2. Corrected Supervised Learning Formulation

### Input & Target
- **Input $X \in [0, 1]^{256 \times 256}$:** Attacked, watermarked normalized $I$-channel (stored as `float16` memory-mapped arrays in `data/ann/{train,val,test}/inputs.npy`).
- **Label $Y \in \{0, 1\}^{32 \times 32}$:** Original $32 \times 32$ binary watermark prior to Arnold Cat Map and Catalan permutation (stored as `uint8` in `data/ann/{train,val,test}/labels.npy`).

### Strict Host-Level & Watermark-Level Split Isolation
1. **Host De-duplication & Splitting (`core/data_sources.py`):**
   - All $1,100$ files are grouped by their 4-digit DIV2K host prefix into **800 unique hosts**.
   - Using `seed = 42`, hosts are partitioned $70\% / 15\% / 15\%$ into:
     - **Train:** `560` unique hosts (`2,240` samples at 4 samples/host)
     - **Validation:** `120` unique hosts (`360` samples at 3 samples/host)
     - **Test:** `120` unique hosts (`360` samples at 3 samples/host)
   - Measured cross-split host overlap: `train ∩ val = 0`, `train ∩ test = 0`, `val ∩ test = 0`.
2. **Disjoint Watermark IDs:**
   - **Train watermark IDs:** sampled from $[10000, 50000)$
   - **Validation watermark IDs:** sampled from $[50000, 60000)$
   - **Test watermark IDs:** sampled from $[60000, 70000)$ plus canonical ID `0` (`PW26_PAC_01`), which is **never seen during training or validation**.

### Attack Distribution in Training Data
Each generated sample passes through the full pipeline (Arnold 10 iters $\to$ Catalan 5 iters, key 7 $\to$ $8 \times 8$ mosaic $\to$ adaptive embedding with $\alpha_{\text{base}} = 0.012, \gamma = 2.0$) followed by a randomized attack:
- `clean` ($12\%$)
- `crop_random` ($15\%$, area removal uniform in $[5\%, 55\%]$)
- `crop_center` ($8\%$, area removal in $\{10\%, 25\%, 50\%\}$)
- `jpeg` ($17\%$, quality $Q \in [45, 95]$)
- `noise` ($13\%$, Gaussian $\sigma \in [0.005, 0.06]$)
- `blur` ($9\%$, Gaussian $3 \times 3, \sigma \in [0, 1]$)
- `median` ($8\%$, $3 \times 3$ median filter)
- `combined` ($18\%$, random crop $[5\%, 40\%]$ followed by JPEG $Q \in [55, 95]$, Gaussian noise $\sigma \in [0.005, 0.04]$, or Gaussian blur $3 \times 3$)

### Why Multi-Copy Collusion Is Evaluated Rather Than Used as a Training Label
In Type-I averaging collusion, $N$ differently watermarked copies $I'_k = I + \alpha \cdot (M_k - 0.5)$ of the same host are averaged:
$$\bar{I} = I + \frac{\alpha}{N}\sum_{k=1}^{N}(M_k - 0.5) + \eta$$
By symmetry, $\bar{I}$ contains all $N$ watermarks with equal weight $1/N$; no function of $\bar{I}$ alone can know *which* of the $N$ colluders was arbitrarily labelled "the victim" during training. Including high-$N$ collusion pairs with a single victim label during training simply injects $(N-1)/N$ label noise. Instead, the ANN is trained to be a high-sensitivity single-image blind detector and its empirical correlation and detection rate under $N \in \{2, 5, 10, 20, 50, 100\}$ colluders are measured on the held-out test set.

---

## 3. Loss Function, Optimizer & Training Protocol

- **Loss (`training/losses.py`):**
  $$\mathcal{L}(\hat{Z}, Y) = \operatorname{BCEWithLogitsLoss}(\hat{Z}, Y) + 0.1 \cdot \bigl(1 - \operatorname{soft\_NC}(\sigma(\hat{Z}), Y)\bigr)$$
- **Optimizer:** `AdamW` ($\text{lr} = 3 \times 10^{-3}$, weight decay $= 10^{-4}$), gradient norm clipping at $5.0$.
- **Scheduler:** `ReduceLROnPlateau` monitoring validation NC (`mode="max"`, factor $= 0.5$, patience $= 2$).
- **Early Stopping & Checkpointing:** Best model selected strictly by **maximum validation NC** (`models/checkpoints/best_model.pth`); latest state saved after every epoch (`models/checkpoints/last_model.pth`) to support `--resume`.
