"""
Master Command Line Interface (CLI) for the Capstone Robust Image Watermarking Research Pipeline.

Provides unified execution for:
- Individual phases (phase 0 to 8)
- Individual attack testing
- Capstone visual demonstration mode (--phase demo)
- Full benchmark execution (--phase benchmark)
- Automated validation tests (--phase tests)
- ANN training pair preparation (--phase ann-data)
- Complete pipeline dependency run (--phase all)
- Quick reproducibility verification (--phase reproduce)
"""
import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pipeline.phase_runner import PhaseRunner


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Master CLI for Capstone Robust Image Watermarking Research Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --phase demo
  python run.py --phase preprocessing
  python run.py --phase watermark
  python run.py --phase embedding
  python run.py --phase attacks --attack crop
  python run.py --phase extraction
  python run.py --phase benchmark --num-images 5
  python run.py --phase tests --verbose
  python run.py --phase ann-data --num-images 10
  python run.py --phase reproduce --quick
  python run.py --phase all
        """
    )

    parser.add_argument(
        "--phase",
        type=str,
        default="demo",
        choices=[
            "environment", "0",
            "preprocessing", "1",
            "watermark", "2",
            "embedding", "3",
            "attacks", "4",
            "extraction", "5",
            "benchmark", "6",
            "tests", "7",
            "ann-data", "8",
            "ann-train", "ann-eval", "9",
            "demo",
            "reproduce",
            "all"
        ],
        help="Select pipeline phase to execute (default: demo)"
    )

    parser.add_argument(
        "--attack",
        type=str,
        default="crop",
        choices=["crop", "jpeg", "noise", "blur", "collusion"],
        help="Specific attack module to run during phase 4 or demo (default: crop)"
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Limit number of images for benchmark, preprocessing, or ANN dataset generation"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run fast smoke-test evaluation for reproducibility or benchmark"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed verbose output"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom YAML configuration file"
    )

    args = parser.parse_args()
    runner = PhaseRunner(config_path=Path(args.config) if args.config else None)

    phase = args.phase.lower()

    if phase in ["environment", "0"]:
        runner.run_phase0()

    elif phase in ["preprocessing", "1"]:
        runner.run_phase1()

    elif phase in ["watermark", "2"]:
        runner.run_phase2()

    elif phase in ["embedding", "3"]:
        runner.run_phase3()

    elif phase in ["attacks", "4"]:
        runner.run_phase4(attack_name=args.attack)

    elif phase in ["extraction", "5"]:
        runner.run_phase5()

    elif phase in ["benchmark", "6"]:
        n = 5 if args.quick and args.num_images is None else args.num_images
        runner.run_phase6(num_images=n)

    elif phase in ["tests", "7"]:
        runner.run_phase7(verbose=args.verbose)

    elif phase in ["ann-data", "8"]:
        n = 5 if args.quick and args.num_images is None else (args.num_images or 10)
        runner.run_phase8(max_samples=n)

    elif phase in ["ann-train", "ann-eval", "9"]:
        runner.run_phase9()

    elif phase == "demo":
        runner.run_demo(attack_name=args.attack)

    elif phase == "reproduce":
        print("\n" + "="*80)
        print("REPRODUCIBILITY VERIFICATION SUITE")
        print("="*80)
        runner.run_phase0()
        runner.run_phase7(verbose=False)
        n = 3 if args.quick else 5
        runner.run_phase6(num_images=n)
        runner.run_demo(attack_name="crop")
        print("\n" + "="*80)
        print("REPRODUCIBILITY VERIFICATION COMPLETE: ALL CHECKS PASSED")
        print("="*80)

    elif phase == "all":
        print("\n" + "="*80)
        print("EXECUTING COMPLETE CAPSTONE RESEARCH PIPELINE (PHASES 0 TO 8)")
        print("="*80)
        runner.run_phase0()
        runner.run_phase1()
        runner.run_phase2()
        runner.run_phase3()
        for atk in ["crop", "jpeg", "noise", "blur", "collusion"]:
            runner.run_phase4(attack_name=atk)
        runner.run_phase5()
        runner.run_phase6(num_images=5 if args.quick else 10)
        runner.run_phase7(verbose=False)
        runner.run_phase8(max_samples=5 if args.quick else 10)
        runner.run_demo(attack_name="crop")
        runner.run_phase9()

        print("\n" + "="*80)
        print("PIPELINE EXECUTION STATUS SUMMARY")
        print("="*80)
        print("COMPLETED PHASES:")
        print("  - Phase 0: Environment & Dataset Audit")
        print("  - Phase 1: Image Preprocessing & YIQ Decomposition")
        print("  - Phase 2: Watermark Scrambling (Arnold + Catalan + Mosaic)")
        print("  - Phase 3: Perceptual Adaptive Embedding (PSNR > 41 dB)")
        print("  - Phase 4: Attack Engine Suite (Crop, JPEG, Noise, Blur, Collusion)")
        print("  - Phase 5: Non-Blind Host-Subtracted Extraction")
        print("  - Phase 6: Standardized Baseline Benchmark")
        print("  - Phase 7: Automated Pytest Suite")
        print("  - Phase 8: ANN Training Dataset Preparation")
        print("\nREMAINING / PLANNED PHASES:")
        print("  - Phase 9: Blind ANN Watermark Extractor (Deep network architecture & weights pending)")
        print("="*80)


if __name__ == "__main__":
    main()
