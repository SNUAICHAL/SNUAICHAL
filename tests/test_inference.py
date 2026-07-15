from pathlib import Path

from snuaichal.inference import build_parser


def test_inference_accepts_a_local_lora_adapter() -> None:
    args = build_parser().parse_args(
        [
            "--adapter-path",
            "outputs/run/final",
            "--validation-fraction",
            "0.05",
            "--metrics-output",
            "outputs/metrics.json",
        ]
    )

    assert args.adapter_path == Path("outputs/run/final")
    assert args.validation_fraction == 0.05
    assert args.metrics_output == Path("outputs/metrics.json")
    assert args.clean_validation is True
    assert args.max_new_tokens == 16
    assert args.min_pixels == 56 * 28 * 28
    assert args.max_pixels == 256 * 28 * 28
