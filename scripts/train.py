"""Training entry point for the zero-shot baseline.

The official baseline performs no fine-tuning and therefore creates no trained
weights. This executable file records that fact explicitly for reproducibility.
"""


def main() -> None:
    print("Zero-shot baseline: no training or fine-tuning is performed.")


if __name__ == "__main__":
    main()

