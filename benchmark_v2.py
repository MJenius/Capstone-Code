"""
Benchmarking V2 CLI wrapper.

Executes the standardized PreANNBenchmarker from benchmark.py.
"""
import argparse
from benchmark import PreANNBenchmarker


def main():
    parser = argparse.ArgumentParser(description="Run Pre-ANN Benchmarking Suite")
    parser.add_argument("--num_images", type=int, default=None, help="Number of test images to evaluate (default: all in test.txt)")
    args = parser.parse_args()

    benchmarker = PreANNBenchmarker()
    benchmarker.run_benchmark(num_images=args.num_images)


if __name__ == "__main__":
    main()
