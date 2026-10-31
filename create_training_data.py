"""
Phase-3 / Final ANN Training Dataset Generator (`create_training_data.py`).

Generates leakage-free (input, label) pairs in `data/ann/{train,val,test}`:
- Input:  Attacked watermarked I-channel (256x256)
- Label:  Original 32x32 binary watermark (32x32)
- Splits: Strict host-level separation across 800 unique DIV2K hosts (560 train / 120 val / 120 test)
          with disjoint watermark ID ranges per split.
"""
import argparse
from pipeline.ann_phases import phase_ann_data


def create_dataset(quick: bool = False, samples_per_host: int = 4, max_hosts: int = None, host_dir: str = None):
    return phase_ann_data(quick=quick, samples_per_host=samples_per_host, max_hosts=max_hosts, host_dir=host_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate leakage-free ANN training/validation/test dataset")
    parser.add_argument("--quick", action="store_true", help="Generate quick subset dataset")
    parser.add_argument("--samples-per-host", type=int, default=4, help="Samples per training host (default: 4)")
    parser.add_argument("--max-hosts", type=int, default=None, help="Optional cap on hosts per split")
    parser.add_argument("--host-dir", type=str, default=None, help="Optional custom host image directory")
    args = parser.parse_args()
    create_dataset(
        quick=args.quick,
        samples_per_host=args.samples_per_host,
        max_hosts=args.max_hosts,
        host_dir=args.host_dir,
    )
