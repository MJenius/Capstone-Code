"""
Held-out evaluation of the blind ANN and fair comparison with the non-blind
extractors and the centre-tile baseline.

Information available to each method (reported in every output):

  ann_blind        : attacked image only (+ secret permutation key)          BLIND
  nonblind_pixel   : attacked image + ORIGINAL HOST + crop mask (+ key);
                     residual divided by the pixel-wise alpha map (corrected)  NON-BLIND
  nonblind_base    : as above but divided by alpha_base (legacy approximation) NON-BLIND
  baseline_center  : attacked image + ORIGINAL HOST, single centre 32x32 tile NON-BLIND

Research metrics
----------------
CRR (Cropping Resistance Ratio), per method and crop level p:
    CRR(p) = mean NC under crop p / mean NC on clean images
  = fraction of clean-condition correlation retained after cropping (1 = no loss).

CoRI (Collusion Robustness Index), per method and colluder count N:
    CoRI(N) = P( NC(recovered, victim watermark) > tau )
  i.e. the victim-fingerprint detection rate after an N-copy averaging attack.
  tau = 4 / sqrt(1024) = 0.125, four standard deviations of the NC of two
  independent random 1024-bit patterns (Gaussian approximation; per-test FPR
  ~3e-5).  The empirical false-positive rate of the ANN at tau is measured
  with unwatermarked hosts and unrelated watermarks (negative control).
  Aggregate CoRI = mean of CoRI(N) over the evaluated N values.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from core.data_sources import Host, get_split_hosts, load_host_bgr, load_host_i, make_watermark, sample_watermark_id
from core.evaluation import calculate_psnr, calculate_ssim
from core.hybrid_system import EVAL_ATTACKS, HybridSystem
from core.preprocessing import bgr_to_yiq, normalize_channel, yiq_to_bgr
from training.metrics import bit_metrics

TAU = 4.0 / np.sqrt(1024)
METHOD_INFO = {
    "ann_blind": "BLIND: attacked image only (+ secret key). No host, no mask.",
    "nonblind_pixel": "NON-BLIND: original host + crop mask; pixel-wise alpha compensation (corrected).",
    "nonblind_base": "NON-BLIND: original host + crop mask; alpha_base compensation (legacy).",
    "baseline_center": "NON-BLIND: original host; conventional single centre-tile embedding.",
}


def rgb_roundtrip(host_bgr: np.ndarray, wm_i_norm: np.ndarray) -> Dict[str, np.ndarray]:
    """Write watermarked I back into RGB (uint8 PNG-equivalent) and re-read it blindly."""
    yiq = bgr_to_yiq(host_bgr)
    _, i_min, i_max = normalize_channel(yiq[:, :, 1])
    yiq_w = yiq.copy()
    yiq_w[:, :, 1] = wm_i_norm * (i_max - i_min) + i_min
    bgr_w = yiq_to_bgr(yiq_w)                    # uint8 quantisation happens here
    i_back = normalize_channel(bgr_to_yiq(bgr_w)[:, :, 1])[0]  # receiver-side, own min/max
    return {"bgr": bgr_w, "i_channel": i_back}


def run_comparison(extractor, system: Optional[HybridSystem] = None, n_hosts: Optional[int] = None,
                   attacks: Optional[List[str]] = None, seed: int = 42, include_rgb: bool = True,
                   progress: Optional[Callable[[int, int], None]] = None, hosts: Optional[List[Host]] = None
                   ) -> Dict[str, pd.DataFrame]:
    system = system or HybridSystem()
    attacks = attacks or list(EVAL_ATTACKS)
    hosts = hosts if hosts is not None else get_split_hosts(seed=seed)["test"]
    if n_hosts:
        hosts = hosts[:n_hosts]
    rng = np.random.RandomState(seed + 7_777)
    rows, neg_rows, samples = [], [], []
    for hi, host in enumerate(hosts):
        host_i = load_host_i(host)
        wm_id = 0 if hi == 0 else sample_watermark_id("test", rng)
        wm = make_watermark(wm_id)
        hyb = system.embed(host_i, wm)
        base = system.embed_baseline(host_i, wm)
        fid = {"hybrid": (calculate_psnr(host_i, hyb), calculate_ssim(host_i, hyb)),
               "baseline": (calculate_psnr(host_i, base), calculate_ssim(host_i, base))}
        inputs, meta = [], []
        for ai, atk in enumerate(attacks):
            a_seed = 100_000 * (hi + 1) + 97 * ai + seed
            cc_h = cc_b = None
            extra = {}
            if atk.startswith("collusion_"):
                n = int(atk.split("_")[1])
                ch = system.collusion_copies(host_i, wm, n, seed=a_seed)
                cb = system.collusion_copies(host_i, wm, n, seed=a_seed, baseline=True)
                cc_h, cc_b = ch["copies"], cb["copies"]
                extra = {"colluder_wm_ids": ch["colluder_wm_ids"][:5], "victim_wm_id": wm_id}
            ah = system.apply_attack(atk, hyb, a_seed, collusion_copies=cc_h)
            ab = system.apply_attack(atk, base, a_seed, collusion_copies=cc_b)
            nb_p = system.extract_nonblind(ah["image"], host_i, ah["mask"], "pixel")
            nb_b = system.extract_nonblind(ah["image"], host_i, ah["mask"], "base")
            bl = system.extract_baseline(ab["image"], host_i)
            common = {"host_id": host.host_id, "watermark_id": wm_id, "attack": atk, "attack_seed": a_seed}
            for m, rec, (ps, ss) in (("nonblind_pixel", nb_p["recovered"], fid["hybrid"]),
                                     ("nonblind_base", nb_b["recovered"], fid["hybrid"]),
                                     ("baseline_center", bl, fid["baseline"])):
                rows.append({**common, "method": m, "psnr": ps, "ssim": ss, **bit_metrics(rec, wm)})
            inputs.append(ah["image"]); meta.append((common, ah, nb_p))
        if include_rgb:
            # non-blind on RGB round trip uses the host I-channel recomputed from the same RGB host
            host_i_rgb = normalize_channel(bgr_to_yiq(load_host_bgr(host))[:, :, 1])[0]
            hyb_rgb = system.embed(host_i_rgb, wm)
            rt = rgb_roundtrip(load_host_bgr(host), hyb_rgb)
            nb = system.extract_nonblind(rt["i_channel"], host_i_rgb, None, "pixel")
            common = {"host_id": host.host_id, "watermark_id": wm_id, "attack": "rgb_png_roundtrip", "attack_seed": 0}
            rows.append({**common, "method": "nonblind_pixel", "psnr": calculate_psnr(host_i_rgb, hyb_rgb),
                         "ssim": calculate_ssim(host_i_rgb, hyb_rgb), **bit_metrics(nb["recovered"], wm)})
            inputs.append(rt["i_channel"]); meta.append((common, None, None))
        # blind ANN on all attacked images of this host (batched, no host given)
        preds = extractor.predict_batch(inputs)
        for (common, ah, nb_p), pr in zip(meta, preds):
            ps, ss = fid["hybrid"]
            rows.append({**common, "method": "ann_blind", "psnr": ps, "ssim": ss,
                         "inference_seconds": pr["inference_seconds"], **bit_metrics(pr["binary"], wm)})
            if hi < 4 and ah is not None and common["attack"] in ("clean", "crop_50", "jpeg_50", "collusion_5"):
                samples.append({"host_id": host.host_id, "attack": common["attack"], "attacked": ah["image"],
                                "ann_prob": pr["probability"], "ann_bin": pr["binary"],
                                "nonblind": nb_p["recovered"], "wm": wm})
        # negative control: unwatermarked host + unrelated watermark + OOD unseen watermarks
        pn = extractor.predict_batch([host_i])[0]
        unrelated_id = sample_watermark_id("test", rng)
        unrelated = make_watermark(unrelated_id)
        # Ensure we also test an unrelated odd-ID watermark (zero template bias) and an i.i.d. Bernoulli(0.5) watermark
        unrelated_odd_id = int(rng.randint(30_000, 35_000)) * 2 + 1
        unrelated_odd = make_watermark(unrelated_odd_id)
        unrelated_iid = (rng.rand(32, 32) > 0.5).astype(np.uint8)
        # Out-of-distribution unseen watermark embedded into this test host
        ood_iid_wm = (rng.rand(32, 32) > 0.5).astype(np.uint8)
        ood_inv_wm = (1 - wm).astype(np.uint8)
        hyb_ood_iid = system.embed(host_i, ood_iid_wm)
        hyb_ood_inv = system.embed(host_i, ood_inv_wm)
        p_ood = extractor.predict_batch([hyb_ood_iid, hyb_ood_inv])
        neg_rows.append({
            "host_id": host.host_id,
            "wm_id": int(wm_id),
            "unrelated_wm_id": int(unrelated_id),
            "both_even_template": bool((wm_id % 2 == 0) and (unrelated_id % 2 == 0)),
            "nc_unwatermarked_vs_wm": bit_metrics(pn["binary"], wm)["nc"],
            "nc_clean_pred_vs_unrelated_wm": bit_metrics(preds[0]["binary"], unrelated)["nc"],
            "nc_clean_pred_vs_unrelated_odd_wm": bit_metrics(preds[0]["binary"], unrelated_odd)["nc"],
            "nc_clean_pred_vs_unrelated_iid_wm": bit_metrics(preds[0]["binary"], unrelated_iid)["nc"],
            "nc_gt_wm_vs_unrelated_wm": bit_metrics(wm, unrelated)["nc"],
            "nc_ood_iid_recovery": bit_metrics(p_ood[0]["binary"], ood_iid_wm)["nc"],
            "ber_ood_iid_recovery": bit_metrics(p_ood[0]["binary"], ood_iid_wm)["ber"],
            "nc_ood_inverted_recovery": bit_metrics(p_ood[1]["binary"], ood_inv_wm)["nc"],
            "ber_ood_inverted_recovery": bit_metrics(p_ood[1]["binary"], ood_inv_wm)["ber"],
        })
        if progress:
            progress(hi + 1, len(hosts))
    df = pd.DataFrame(rows)
    return {"records": df, "negative_control": pd.DataFrame(neg_rows), "samples": samples}


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["method", "attack"])
    agg = g.agg(n=("nc", "size"),
                nc_mean=("nc", "mean"), nc_std=("nc", "std"), ber_mean=("ber", "mean"), ber_std=("ber", "std"),
                accuracy_mean=("accuracy", "mean"), precision_mean=("precision", "mean"),
                recall_mean=("recall", "mean"), f1_mean=("f1", "mean"),
                psnr_mean=("psnr", "mean"), psnr_std=("psnr", "std"),
                ssim_mean=("ssim", "mean"), ssim_std=("ssim", "std"),
                detect_rate=("nc", lambda s: float((s > TAU).mean()))).reset_index()
    return agg


def research_metrics(agg: pd.DataFrame) -> Dict:
    out = {"tau": TAU, "CRR": {}, "CoRI": {}, "CoRI_aggregate": {}}
    for m in agg.method.unique():
        a = agg[agg.method == m].set_index("attack")
        if "clean" in a.index and a.loc["clean", "nc_mean"] > 0:
            out["CRR"][m] = {k: float(a.loc[k, "nc_mean"] / a.loc["clean", "nc_mean"])
                             for k in a.index if k.startswith("crop_")}
        col = {k: float(a.loc[k, "detect_rate"]) for k in a.index if k.startswith("collusion_")}
        if col:
            out["CoRI"][m] = col
            out["CoRI_aggregate"][m] = float(np.mean(list(col.values())))
    return out


def evaluate_ann_on_split(extractor, split: str = "test") -> Dict:
    """Metrics of the blind ANN on the stored held-out split (per attack type)."""
    from training.dataset import ANN_DIR
    X = np.load(ANN_DIR / split / "inputs.npy", mmap_mode="r")
    Y = np.load(ANN_DIR / split / "labels.npy")
    meta = pd.read_csv(ANN_DIR / "metadata.csv")
    meta = meta[meta.split == split].sort_values("index").reset_index(drop=True)
    rows = []
    t0 = time.time()
    for s in range(0, len(Y), 32):
        preds = extractor.predict_batch([np.asarray(X[i], np.float32) for i in range(s, min(s + 32, len(Y)))])
        for k, p in enumerate(preds):
            i = s + k
            rows.append({"index": i, "attack_type": meta.attack_type[i], "watermark_id": int(meta.watermark_id[i]),
                         "host_id": meta.host_id[i], **bit_metrics(p["binary"], Y[i])})
    df = pd.DataFrame(rows)
    per = df.groupby("attack_type").agg(n=("nc", "size"), nc_mean=("nc", "mean"), nc_std=("nc", "std"),
                                        ber_mean=("ber", "mean"), ber_std=("ber", "std"),
                                        accuracy_mean=("accuracy", "mean"), f1_mean=("f1", "mean")).reset_index()
    overall = {k: float(df[k].mean()) for k in ("nc", "ber", "accuracy", "precision", "recall", "f1")}
    return {"records": df, "per_attack": per, "overall": overall, "n": len(df),
            "n_hosts": int(df.host_id.nunique()), "seconds": time.time() - t0}
