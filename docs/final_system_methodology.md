# Final Hybrid System Methodology

## 1. End-to-End Mathematical Pipeline

### Stage 1: Host Image Validation, Resizing & YIQ Decomposition (`core/preprocessing.py`)
Given an input RGB/BGR host image $H$, we resize via bicubic interpolation to $(256, 256, 3)$ and apply the NTSC transformation:
$$\begin{bmatrix} Y \\ I \\ Q \end{bmatrix} = \begin{bmatrix} 0.299 & 0.587 & 0.114 \\ 0.596 & -0.274 & -0.322 \\ 0.211 & -0.523 & 0.312 \end{bmatrix} \begin{bmatrix} R \\ G \\ B \end{bmatrix}$$
The chrominance $I$-channel is min–max normalized to $I_{\text{norm}} \in [0, 1]$ (`float32`), storing $(I_{\min}, I_{\max})$ for inverse RGB reconstruction.

### Stage 2: Dual Keyed Watermark Scrambling & $8 \times 8$ Mosaic (`core/watermark.py`)
Given a binary watermark $W \in \{0, 1\}^{32 \times 32}$:
1. **Arnold Cat Map (ACM):** $k_{\text{acm}} = 10$ iterations of the discrete torus map:
   $$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} \bmod 32, \qquad \text{Inverse: } \begin{bmatrix} 2 & -1 \\ -1 & 1 \end{bmatrix} \begin{bmatrix} x' \\ y' \end{bmatrix} \bmod 32$$
2. **Catalan Permutation:** $k_{\text{cat}} = 5$ iterations with secret key $K = 7$. At iteration $t$, Catalan numbers $C_{n + K + t}$ are hashed with `(index, K, t)` via 64-bit `blake2b` to define a collision-free permutation $\pi_t$ of the $1,024$ elements.
3. **$8 \times 8$ Spatial Mosaic:** The transformed $32 \times 32$ array $W_C = \pi_{\text{key}}(W)$ is tiled in an $8 \times 8$ grid (`np.tile(W_C, (8, 8))`) to form $W_{\text{mosaic}} \in \{0, 1\}^{256 \times 256}$ containing $64$ redundant copies.

### Stage 3: Perceptual Luminance-Texture Adaptive Embedding (`core/embedding.py`)
Using a $7 \times 7$ box filter on $I_{\text{norm}}$, local variance $\sigma^2(x,y) = E[I^2] - (E[I])^2$ is converted to a normalized standard deviation mask $M(x,y) \in [0, 1]$:
$$\alpha(x,y) = \alpha_{\text{base}}\bigl(1 + \gamma \, M(x,y)\bigr), \qquad I'(x,y) = \operatorname{clip}\!\Bigl(I_{\text{norm}}(x,y) + \alpha(x,y)\bigl(W_{\text{mosaic}}(x,y) - 0.5\bigr),\ 0,\ 1\Bigr)$$
With $\alpha_{\text{base}} = 0.012, \gamma = 2.0$, $\alpha(x,y) \in [0.012, 0.036]$, achieving mean clean $\text{PSNR} = 41.57\text{ dB}$ and $\text{SSIM} = 0.9965$.

### Stage 4: Extraction Paths

#### Path A — Non-Blind (Host-Subtracted) Extraction (`core/extraction.py`, `core/hybrid_system.py`)
Requires the original unwatermarked host $I_{\text{norm}}$ (and crop mask $K_{\text{crop}}$ if cropped):
1. Compute residual $\hat{D}(x,y) = \frac{I_{\text{attacked}}(x,y) - I_{\text{norm}}(x,y)}{\alpha(x,y)} + 0.5$ (`nonblind_pixel`, exact inverse) or divide by $\alpha_{\text{base}}$ (`nonblind_base`, legacy).
2. Fold $\hat{D}$ into $64$ tiles of $32 \times 32$ and compute mask-weighted average $\hat{W}_C$.
3. Threshold at $0.5$ and apply inverse Catalan followed by inverse Arnold Cat Map.

#### Path B — Blind ANN Extraction (`models/blind_extractor.py`, `training/inference.py`)
Requires **only** the attacked image $I_{\text{attacked}} \in [0, 1]^{256 \times 256}$ (and the fixed key permutation buffer):
1. Condition input via per-image Z-normalization and $5 \times 5$ high-pass residual.
2. Extract multi-scale residual CNN features ($256 \times 256$ and $128 \times 128$).
3. Fold into $64 \times 32 \times 32$ tiles and aggregate using learned per-pixel softmax confidence weights + mean pooling.
4. Decode via $32 \times 32$ residual blocks and gather through the fixed inverse permutation buffer $\pi_{\text{key}}^{-1}$ to output $\hat{P} \in [0, 1]^{32 \times 32}$ and $\hat{W} = (\hat{P} > 0.5)$.

---

## 2. Formal Definitions of Research-Specific Metrics

### Cropping Resistance Ratio ($\text{CRR}$)
For a given cropping area removal fraction $p \in \{0.10, 0.25, 0.50\}$, $\text{CRR}(p)$ measures the fraction of clean-image Normalized Correlation retained after removing fraction $p$ of the image area:
$$\text{CRR}(p) = \frac{\mathbb{E}[\text{NC}(\hat{W}_{\text{crop}(p)}, W)]}{\mathbb{E}[\text{NC}(\hat{W}_{\text{clean}}, W)]}$$
- $\text{CRR}(p) \approx 1.0$ indicates near-complete immunity to cropping of area fraction $p$.
- For a single localized center watermark (`baseline_center`), random cropping removes the center region with probability increasing with $p$, causing $\text{CRR}(0.50) \to 0$.

### Collusion Robustness Index ($\text{CoRI}$)
Under an $N$-colluder Type-I averaging attack (1 victim watermark $W_{\text{victim}}$ + $N-1$ independent watermarks averaged with noise $\sigma_c = 0.01$), let $\tau = \frac{4}{\sqrt{1024}} = 0.125$ be the $4\sigma$ statistical detection threshold against random $1,024$-bit patterns (false-positive probability $\approx 3.2 \times 10^{-5}$ under the null hypothesis $\text{NC} \sim \mathcal{N}(0, 1/1024)$). Then:
$$\text{CoRI}(N) = \Pr\bigl(\text{NC}(\hat{W}^{(N)}, W_{\text{victim}}) > \tau\bigr), \qquad \overline{\text{CoRI}} = \frac{1}{|\mathcal{N}|}\sum_{N \in \{2,5,10,20,50,100\}} \text{CoRI}(N)$$
$\text{CoRI}(N) \in [0, 1]$ represents the empirical probability of successfully tracing the victim's fingerprint after $N$ colluders average their differently watermarked copies.
