import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import run_blocked_low_validation_resume as resume


def test_direct_script_entrypoint_resolves_project_imports() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(resume.ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(resume.ROOT / "scripts" / "run_blocked_low_validation_resume.py"),
            "--help",
        ],
        cwd=resume.ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--poll-seconds" in result.stdout


def test_select_aggregation_uses_exact_match_before_tie_breaks() -> None:
    metrics = {
        "samples": 100,
        "parse_failures": 0,
        "inference_seconds_per_sample": 8.0,
        "aggregation_comparison": {
            "hard": {"vs_hard": {"accuracy": 0.25, "exact_matches": 25}},
            "confidence_tiebreak": {
                "vs_hard": {"accuracy": 0.27, "exact_matches": 27}
            },
            "confidence_weighted": {
                "vs_hard": {"accuracy": 0.26, "exact_matches": 26}
            },
        },
    }

    selected = resume.select_aggregation(metrics)

    assert selected["mode"] == "confidence_tiebreak"
    assert selected["exact_matches"] == 27
    assert selected["accuracy"] == 0.27
    assert selected["parse_failures"] == 0
    assert selected["inference_seconds_per_sample"] == 8.0


def test_mode_specific_validation_metrics_follow_selected_aggregation() -> None:
    metrics = {
        "samples": 2,
        "parse_failures": 0,
        "aggregation_mode": "hard",
        "exact_matches": 1,
        "exact_match": 0.5,
        "non_identity_exact_matches": 0,
        "non_identity_exact_match": 0.0,
        "aggregation_comparison": {
            "confidence_tiebreak": {
                "vs_hard": {"exact_matches": 2, "accuracy": 1.0}
            }
        },
    }
    audit_rows = [
        {
            "Id": "a",
            "reference": [2, 1, 3, 4],
            "no_ordering": False,
            "aggregations": {
                "confidence_tiebreak": {"answer": [2, 1, 3, 4]}
            },
        },
        {
            "Id": "b",
            "reference": [1, 2, 3, 4],
            "no_ordering": True,
            "aggregations": {
                "confidence_tiebreak": {"answer": [1, 2, 3, 4]}
            },
        },
    ]

    selected = resume.validation_metrics_for_aggregation(
        metrics, audit_rows, "confidence_tiebreak"
    )

    assert selected["aggregation_mode"] == "confidence_tiebreak"
    assert selected["exact_matches"] == 2
    assert selected["exact_match"] == 1.0
    assert selected["non_identity_exact_matches"] == 1
    assert selected["non_identity_exact_match"] == 1.0

    candidate = resume.build_8b_candidate(
        best={"step": 4292},
        best_adapter=Path("outputs/qwen3-vl-8b-aug/checkpoint-4292"),
        model_spec={
            "repository": "Qwen/Qwen3-VL-8B-Instruct",
            "verified_model_tree_sha256": "tree-sha",
        },
        tta4_metrics=metrics,
        tta4_audit=audit_rows,
        selected_mode="confidence_tiebreak",
    )
    assert candidate["aggregation_mode"] == "confidence_tiebreak"
    assert candidate["validation"]["exact_matches"] == 2
    assert "confidence_tiebreak" in candidate["name"]


def test_stratified_subset_is_exact_deterministic_and_immutable(tmp_path) -> None:
    rows = [
        {
            "Id": f"identity-{index}",
            "Answer": "[1, 2, 3, 4]",
            "No_ordering": "True",
        }
        for index in range(30)
    ] + [
        {
            "Id": f"other-{index}",
            "Answer": "[2, 1, 3, 4]",
            "No_ordering": "False",
        }
        for index in range(70)
    ]
    validation_ids = [row["Id"] for row in rows]

    manifest = resume.build_stratified_subset_manifest(
        rows,
        validation_ids=validation_ids,
        size=20,
        seed=42,
        source_manifest_sha256="abc123",
    )

    assert len(manifest["validation_ids"]) == 20
    assert manifest["strata"] == {
        "[1, 2, 3, 4]|true": {"available": 30, "selected": 6},
        "[2, 1, 3, 4]|false": {"available": 70, "selected": 14},
    }
    assert manifest == resume.build_stratified_subset_manifest(
        rows,
        validation_ids=validation_ids,
        size=20,
        seed=42,
        source_manifest_sha256="abc123",
    )

    path = tmp_path / "manifest.json"
    resume.write_immutable_json(path, manifest)
    resume.write_immutable_json(path, manifest)
    with pytest.raises(ValueError, match="immutable"):
        resume.write_immutable_json(path, {**manifest, "seed": 7})


def test_paired_decision_requires_gain_or_material_secondary_improvement() -> None:
    eight = {
        "exact_match": 0.25,
        "parse_failures": 2,
        "non_identity_exact_match": 0.20,
    }

    gain = resume.paired_decision(
        eight,
        {
            "exact_match": 0.29,
            "parse_failures": 2,
            "non_identity_exact_match": 0.20,
        },
    )
    assert gain["continue_to_full_validation"] is True
    assert gain["reason"] == "exact_match_gain_at_least_0.03"

    secondary = resume.paired_decision(
        eight,
        {
            "exact_match": 0.24,
            "parse_failures": 1,
            "non_identity_exact_match": 0.24,
        },
    )
    assert secondary["continue_to_full_validation"] is True
    assert secondary["reason"] == "within_0.03_with_strict_secondary_improvement"

    stop = resume.paired_decision(
        eight,
        {
            "exact_match": 0.21,
            "parse_failures": 0,
            "non_identity_exact_match": 0.30,
        },
    )
    assert stop["continue_to_full_validation"] is False
    assert stop["reason"] == "paired_gate_not_met"


def test_inference_command_is_local_validation_only() -> None:
    command = resume.build_inference_command(
        model_path=resume.MODEL_27B,
        adapter_path=None,
        validation_manifest=resume.RESUME_ROOT / "paired-96-manifest.json",
        tta=1,
        model_family="qwen3_5",
        model_revision=resume.MODEL_27B_REVISION,
        precision="nf4",
    )

    assert "--test-csv" in command
    assert "data/train.csv" in command
    assert "--validation-manifest" in command
    assert "--precision" in command
    assert "nf4" in command
    assert "--adapter-path" not in command
    assert all("kaggle" not in value.lower() for value in command)
    assert all("submit" not in value.lower() for value in command)


def test_final_test_inference_stage_freezes_identity_and_validates_819_rows(
    tmp_path: Path,
) -> None:
    test_csv = tmp_path / "test.csv"
    image_dir = tmp_path / "test"
    image_dir.mkdir()
    ids = [f"test-{index:04d}" for index in range(819)]
    with test_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Id"])
        writer.writeheader()
        writer.writerows({"Id": sample_id} for sample_id in ids)
    adapter = tmp_path / "checkpoint-3000"
    adapter.mkdir()
    model = tmp_path / "model"
    model.mkdir()
    selection = {
        "model_path": model,
        "model_repository": "Qwen/Qwen3-VL-8B-Instruct",
        "model_family": "qwen3_vl",
        "model_revision": "revision-1",
        "model_manifest": tmp_path / "model-manifest.json",
        "verified_model_tree_sha256": "verified-model-tree",
        "adapter_path": adapter,
        "precision": "nf4",
        "image_size": 512,
        "tta_orders": [
            [1, 2, 3, 4],
            [2, 3, 4, 1],
            [3, 4, 1, 2],
            [4, 1, 2, 3],
        ],
        "aggregation_mode": "hard",
        "fallback_policy": "identity",
        "seed": 42,
    }
    stage = resume.build_final_test_inference_stage(
        selection=selection,
        test_csv=test_csv,
        image_dir=image_dir,
        test_input_identity={"tree_sha256": "test-inputs"},
    )

    assert stage["uses_cuda"] is True
    expected_selection = {
        **selection,
        "model_path": resume._project_path(selection["model_path"]),
        "model_manifest": resume._project_path(selection["model_manifest"]),
        "adapter_path": resume._project_path(adapter),
    }
    assert stage["provenance_context"]["selected_model"] == expected_selection
    command = stage["command"]
    for flag in (
        "--model-path",
        "--model-repository",
        "--model-family",
        "--model-revision",
        "--model-manifest",
        "--adapter-path",
        "--precision",
        "--image-size",
        "--tta",
        "--tta-orders-json",
        "--aggregation-mode",
        "--fallback-policy",
        "--seed",
        "--output",
        "--audit-log",
        "--metrics-output",
    ):
        assert flag in command

    attempt = tmp_path / "attempt-001"
    attempt.mkdir()
    with (attempt / "submission.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Id", "Answer"])
        writer.writeheader()
        writer.writerows(
            {"Id": sample_id, "Answer": "[1, 2, 3, 4]"} for sample_id in ids
        )
    (attempt / "audit.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "Id": sample_id,
                    "answer": [1, 2, 3, 4],
                    "parse_ok": True,
                    "valid_tta_views": 4,
                    "aggregation_mode": "hard",
                    "views": [{}, {}, {}, {}],
                }
            )
            + "\n"
            for sample_id in ids
        ),
        encoding="utf-8",
    )
    (attempt / "metrics.json").write_text(
        json.dumps(
            {
                "samples": 819,
                "parse_failures": 0,
                "tta_views": 4,
                "aggregation_mode": "hard",
                "inference_seconds_per_sample": 1.0,
                "estimated_test_seconds": 819.0,
                "peak_vram_mib": 1024.0,
                "model_precision": "nf4",
                "model_family": "qwen3_vl",
                "detected_model_family": "qwen3_vl",
                "declared_model_family": "qwen3_vl",
                "model_revision": "revision-1",
                "base_model_path": str(model),
                "adapter_path": str(adapter),
                "runtime_state": {
                    "quantization_applied": True,
                    "adapter_loaded": True,
                    "precision": "nf4",
                    "cuda_available": True,
                    "model_eval": True,
                    "use_cache": True,
                },
            }
        ),
        encoding="utf-8",
    )
    validator = stage["validator"]
    assert validator(attempt) == []

    audit_rows = [
        json.loads(line)
        for line in (attempt / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    audit_rows[0]["answer"] = [2, 1, 3, 4]
    (attempt / "audit.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in audit_rows),
        encoding="utf-8",
    )
    assert any("submission answer disagrees with audit" in error for error in validator(attempt))
    audit_rows[0]["answer"] = [1, 2, 3, 4]
    (attempt / "audit.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in audit_rows),
        encoding="utf-8",
    )

    rows = list(csv.DictReader((attempt / "submission.csv").open(encoding="utf-8")))
    rows[1]["Id"] = rows[0]["Id"]
    with (attempt / "submission.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Id", "Answer"])
        writer.writeheader()
        writer.writerows(rows)
    errors = validator(attempt)
    assert any("duplicate" in error or "ordered IDs" in error for error in errors)


def test_final_model_selection_uses_validated_metrics_and_freezes_candidate() -> None:
    selected = resume.select_final_model(
        [
            {
                "name": "8b",
                "model_path": Path("models/8b"),
                "model_family": "qwen3_vl",
                "model_revision": "rev-8b",
                "adapter_path": Path("outputs/checkpoint-3000"),
                "precision": "nf4",
                "image_size": 512,
                "tta_orders": [[1, 2, 3, 4]],
                "aggregation_mode": "hard",
                "fallback_policy": "identity",
                "seed": 42,
                "validation": {
                    "samples": 954,
                    "exact_matches": 250,
                    "exact_match": 250 / 954,
                    "non_identity_exact_matches": 200,
                    "non_identity_exact_match": 0.25,
                    "parse_failures": 0,
                    "inference_seconds_per_sample": 2.0,
                },
            },
            {
                "name": "27b",
                "model_path": Path("models/27b"),
                "model_family": "qwen3_5",
                "model_revision": "rev-27b",
                "adapter_path": None,
                "precision": "nf4",
                "image_size": 512,
                "tta_orders": [[1, 2, 3, 4]],
                "aggregation_mode": "hard",
                "fallback_policy": "identity",
                "seed": 42,
                "validation": {
                    "samples": 954,
                    "exact_matches": 251,
                    "exact_match": 251 / 954,
                    "non_identity_exact_matches": 201,
                    "non_identity_exact_match": 0.26,
                    "parse_failures": 1,
                    "inference_seconds_per_sample": 5.0,
                },
            },
        ]
    )

    assert selected["name"] == "27b"
    assert selected["model_path"] == Path("models/27b")
    assert selected["adapter_path"] is None
    with pytest.raises(ValueError, match="954"):
        resume.select_final_model(
            [{**selected, "validation": {**selected["validation"], "samples": 953}}]
        )


def test_runner_rejects_boolean_submission_frame_numbers() -> None:
    assert resume._valid_permutation([1, 2, 3, 4]) is True
    assert resume._valid_permutation([True, 2, 3, 4]) is False
    assert resume._valid_permutation([1.0, 2, 3, 4]) is False


def test_runner_builds_sanitized_child_environment_without_mutating_parent(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "foreign-site-packages")
    monkeypatch.setenv("PYTHONHOME", "foreign-python")
    monkeypatch.setenv("KAGGLE_KEY", "secret")
    monkeypatch.setenv("GH_TOKEN", "secret")
    monkeypatch.setenv("GIT_ASKPASS", "secret-helper")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("NVIDIA_API_KEY", "secret")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
    monkeypatch.setenv("HTTPS_PROXY", "https://user:secret@proxy.example:443")
    monkeypatch.setenv("CUDA_PATH", r"C:\CUDA")
    before = dict(os.environ)

    child = resume.build_ml_child_environment()

    assert dict(os.environ) == before
    assert child["PYTHONPATH"] == str(resume.ROOT / "src")
    assert "PYTHONHOME" not in child
    assert "KAGGLE_KEY" not in child
    assert "GH_TOKEN" not in child
    assert "GIT_ASKPASS" not in child
    assert "AWS_SECRET_ACCESS_KEY" not in child
    assert "NVIDIA_API_KEY" not in child
    assert "HTTPS_PROXY" not in child
    assert child["NVIDIA_VISIBLE_DEVICES"] == "all"
    assert child["CUDA_PATH"] == r"C:\CUDA"
    assert child["PATH"] == before["PATH"]


def test_acquisition_environment_uses_private_token_free_hf_home(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(resume, "RESUME_ROOT", tmp_path)
    monkeypatch.setenv("HF_HOME", r"C:\Users\sky_m\.cache\huggingface")
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "secret")
    before = dict(os.environ)

    child = resume.build_acquisition_child_environment()

    assert dict(os.environ) == before
    assert child["HF_HOME"] == str(tmp_path / "hf-home")
    assert child["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert "HF_TOKEN" not in child
    assert "HUGGING_FACE_HUB_TOKEN" not in child


def test_paired_audit_comparison_reports_directional_changes() -> None:
    eight = [
        {
            "Id": "a",
            "answer": [1],
            "reference": [1],
            "no_ordering": True,
            "parse_ok": True,
        },
        {
            "Id": "b",
            "answer": [1],
            "reference": [2],
            "no_ordering": False,
            "parse_ok": True,
        },
        {
            "Id": "c",
            "answer": [3],
            "reference": [3],
            "no_ordering": False,
            "parse_ok": True,
        },
    ]
    twenty_seven = [
        {
            "Id": "a",
            "answer": [2],
            "reference": [1],
            "no_ordering": True,
            "parse_ok": True,
        },
        {
            "Id": "b",
            "answer": [2],
            "reference": [2],
            "no_ordering": False,
            "parse_ok": True,
        },
        {
            "Id": "c",
            "answer": [3],
            "reference": [3],
            "no_ordering": False,
            "parse_ok": True,
        },
    ]

    comparison = resume.compare_paired_audit_rows(eight, twenty_seven)

    assert comparison == {
        "samples": 3,
        "prediction_changes": 2,
        "corrected_by_27b": 1,
        "worsened_by_27b": 1,
        "both_correct": 1,
        "both_wrong": 0,
        "net_gain_27b": 0,
    }


def test_paired_audit_comparison_rejects_different_id_order() -> None:
    eight = [{"Id": "a", "answer": [1], "reference": [1], "parse_ok": True}]
    twenty_seven = [
        {"Id": "b", "answer": [1], "reference": [1], "parse_ok": True}
    ]

    with pytest.raises(ValueError, match="same ordered IDs"):
        resume.compare_paired_audit_rows(eight, twenty_seven)


def test_parse_failure_fallback_never_earns_validation_credit() -> None:
    fallback = {
        "Id": "identity",
        "answer": [1, 2, 3, 4],
        "reference": [1, 2, 3, 4],
        "parse_ok": False,
    }
    valid_wrong = {
        "Id": "identity",
        "answer": [2, 1, 3, 4],
        "reference": [1, 2, 3, 4],
        "parse_ok": True,
    }

    comparison = resume.compare_paired_audit_rows([fallback], [valid_wrong])

    assert comparison["both_wrong"] == 1
    assert comparison["worsened_by_27b"] == 0


def test_paired_gate_accepts_any_strict_non_identity_improvement() -> None:
    decision = resume.paired_decision(
        {
            "exact_match": 0.25,
            "parse_failures": 0,
            "non_identity_exact_match": 0.20,
        },
        {
            "exact_match": 0.24,
            "parse_failures": 0,
            "non_identity_exact_match": 0.201,
        },
    )

    assert decision["continue_to_full_validation"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("training_loss", float("nan")),
        ("training_loss", float("inf")),
        ("training_loss", -0.1),
        ("training_loss", 0.0),
        ("learning_rate", float("nan")),
        ("learning_rate", -0.1),
        ("epoch", 0.0),
        ("train_runtime_seconds", 0.0),
        ("train_steps_per_second", float("nan")),
        ("seconds_per_optimizer_step", -1.0),
        ("peak_vram_bytes", 0),
        ("peak_vram_bytes", resume.PHYSICAL_VRAM_BYTES + 1),
    ],
)
def test_training_validator_rejects_impossible_smoke_metrics(
    tmp_path, field, value
) -> None:
    output = tmp_path / "training"
    checkpoint = output / "checkpoint-2"
    final = output / "final"
    checkpoint.mkdir(parents=True)
    final.mkdir(parents=True)
    summary = {
        "global_step": 2,
        "training_loss": 0.5,
        "train_runtime_seconds": 100.0,
        "train_steps_per_second": 0.02,
        "seconds_per_optimizer_step": 50.0,
        "peak_vram_bytes": 20 * 1024**3,
    }
    summary[field] = value
    (output / "training_summary.json").write_text(json.dumps(summary))
    (output / "model_manifest.json").write_text(
        json.dumps(
            {
                "load_in_4bit": True,
                "trainable_parameters": 100,
                "total_parameters": 1_000,
                "model_path": "models/Qwen3.5-27B",
                "lora_rank": 8,
            }
        )
    )
    (output / "schedule.json").write_text(json.dumps({"stop_after_steps": 2}))
    for path in [
        final / "adapter_config.json",
        final / "adapter_model.safetensors",
        checkpoint / "trainer_state.json",
        checkpoint / "adapter_config.json",
        checkpoint / "adapter_model.safetensors",
        checkpoint / "optimizer.pt",
        checkpoint / "scheduler.pt",
        checkpoint / "rng_state.pth",
    ]:
        path.write_bytes(b"nonempty")

    errors = resume.training_validator(2)(tmp_path)

    assert errors


def test_run_stage_reuses_only_exact_provenance(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(resume, "RESUME_ROOT", tmp_path)
    monkeypatch.setattr(resume, "STAGES_ROOT", tmp_path / "stages")
    monkeypatch.setattr(resume, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(resume, "REGISTRY_PATH", tmp_path / "registry.jsonl")

    calls = []
    child_environments = []

    def fake_run(command, attempt_dir, **kwargs):
        calls.append(list(command))
        child_environments.append(kwargs.get("environment"))
        attempt_dir.mkdir(parents=True)
        (attempt_dir / "exit-code.txt").write_text("0\n")
        (attempt_dir / "result.txt").write_text("ok\n")
        return 0

    monkeypatch.setattr(resume.durable, "run_command", fake_run)

    def validator(attempt):
        return [] if (attempt / "result.txt").is_file() else ["missing"]

    first = resume.run_stage(
        "stage",
        ["tool", "--seed", "42", "{attempt_dir}/result.txt"],
        validator=validator,
        poll_seconds=0.01,
        provenance_context={"seed": 42, "expected_output_schema": 1},
    )
    reused = resume.run_stage(
        "stage",
        ["tool", "--seed", "42", "{attempt_dir}/result.txt"],
        validator=validator,
        poll_seconds=0.01,
        provenance_context={"seed": 42, "expected_output_schema": 1},
    )
    (first / "result.txt").write_text("tampered\n")
    artifact_manifest = json.loads(
        (first / "artifact-manifest.json").read_text(encoding="utf-8")
    )
    result_record = next(
        record for record in artifact_manifest["files"] if record["path"] == "result.txt"
    )
    result_record["size"] = (first / "result.txt").stat().st_size
    result_record["sha256"] = resume.sha256_file(first / "result.txt")
    (first / "artifact-manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tampered = resume.run_stage(
        "stage",
        ["tool", "--seed", "42", "{attempt_dir}/result.txt"],
        validator=validator,
        poll_seconds=0.01,
        provenance_context={"seed": 42, "expected_output_schema": 1},
    )
    changed = resume.run_stage(
        "stage",
        ["tool", "--seed", "7", "{attempt_dir}/result.txt"],
        validator=validator,
        poll_seconds=0.01,
        provenance_context={"seed": 7, "expected_output_schema": 1},
    )

    assert reused == first
    assert tampered != first
    assert changed != tampered
    assert len(calls) == 3
    assert all(environment is not None for environment in child_environments)
    assert all("KAGGLE_KEY" not in environment for environment in child_environments)


def test_run_stage_finalizes_external_output_before_recording_identity(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(resume, "RESUME_ROOT", tmp_path)
    monkeypatch.setattr(resume, "STAGES_ROOT", tmp_path / "stages")
    monkeypatch.setattr(resume, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(resume, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    external = tmp_path / "model"
    calls = []

    def fake_run(_command, attempt_dir, **_kwargs):
        calls.append(1)
        attempt_dir.mkdir(parents=True)
        external.mkdir(exist_ok=True)
        (external / "config.json").write_text("{}")
        (attempt_dir / "exit-code.txt").write_text("0\n")
        return 0

    monkeypatch.setattr(resume.durable, "run_command", fake_run)

    def finalize() -> None:
        (external / "SNAPSHOT_REVISION").write_text("revision=pinned\n")

    first = resume.run_stage(
        "download",
        ["hf", "download"],
        validator=lambda _attempt: [],
        poll_seconds=0.01,
        provenance_context={"revision": "pinned"},
        external_outputs=(external,),
        post_success=finalize,
    )
    reused = resume.run_stage(
        "download",
        ["hf", "download"],
        validator=lambda _attempt: [],
        poll_seconds=0.01,
        provenance_context={"revision": "pinned"},
        external_outputs=(external,),
        post_success=finalize,
    )

    status = json.loads((tmp_path / "status.json").read_text())
    recorded = status["experiments"]["download"]["external_output_identities"]
    assert first == reused
    assert recorded == [resume._path_identity(external)]
    assert len(calls) == 1


def test_run_stage_exception_records_terminal_failed_state(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(resume, "RESUME_ROOT", tmp_path)
    monkeypatch.setattr(resume, "STAGES_ROOT", tmp_path / "stages")
    monkeypatch.setattr(resume, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(resume, "REGISTRY_PATH", tmp_path / "registry.jsonl")

    def explode(command, attempt_dir, **kwargs):
        attempt_dir.mkdir(parents=True)
        raise RuntimeError("boom")

    monkeypatch.setattr(resume.durable, "run_command", explode)
    with pytest.raises(RuntimeError, match="boom"):
        resume.run_stage(
            "stage",
            ["tool"],
            validator=lambda attempt: [],
            poll_seconds=0.01,
            provenance_context={"seed": 42},
        )

    status = json.loads((tmp_path / "status.json").read_text())
    record = status["experiments"]["stage"]
    assert record["status"] == "failed"
    assert record["ended_at"]
    assert "active_experiment" not in status
    assert (tmp_path / "stages" / "stage" / "attempt-001" / "traceback.txt").is_file()


def test_content_addressed_reports_never_overwrite(tmp_path) -> None:
    first = resume.write_versioned_report(tmp_path, "report", {"value": 1})
    again = resume.write_versioned_report(tmp_path, "report", {"value": 1})
    second = resume.write_versioned_report(tmp_path, "report", {"value": 2})

    assert first == again
    assert first != second
    assert json.loads(first.read_text()) == {"value": 1}
    assert json.loads(second.read_text()) == {"value": 2}


def test_preserved_manifest_and_selected_images_are_semantically_hashed(tmp_path) -> None:
    (tmp_path / "artifact.bin").write_bytes(b"artifact")
    manifest = tmp_path / "preserved.json"
    manifest.write_text(
        json.dumps(
            {
                "file_count": 1,
                "files": [
                    {
                        "path": "artifact.bin",
                        "size_bytes": 8,
                        "sha256": resume.sha256_file(tmp_path / "artifact.bin"),
                    }
                ],
            }
        )
    )
    verified = resume.verify_preserved_manifest(manifest, root=tmp_path)
    assert verified["verified_file_count"] == 1
    (tmp_path / "artifact.bin").write_bytes(b"mutation")
    with pytest.raises(ValueError, match="preserved"):
        resume.verify_preserved_manifest(manifest, root=tmp_path)

    csv_path = tmp_path / "train.csv"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    names = [f"x_{index}.jpg" for index in range(4)]
    (image_dir / "x").mkdir()
    for index, name in enumerate(names):
        (image_dir / "x" / name).write_bytes(f"image-{index}".encode())
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Id",
                "Input_1",
                "Input_2",
                "Input_3",
                "Input_4",
                "Sentence",
                "Answer",
                "No_ordering",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Id": "x",
                "Input_1": names[0],
                "Input_2": names[1],
                "Input_3": names[2],
                "Input_4": names[3],
                "Sentence": "s",
                "Answer": "[1, 2, 3, 4]",
                "No_ordering": "True",
            }
        )
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"validation_ids": ["x"]}))
    before = resume.validation_input_identity(csv_path, image_dir, split)
    (image_dir / "x" / names[0]).write_bytes(b"changed")
    after = resume.validation_input_identity(csv_path, image_dir, split)
    assert before["tree_sha256"] != after["tree_sha256"]


def test_training_input_manifest_binds_selected_rows_images_and_split(
    monkeypatch, tmp_path
) -> None:
    from PIL import Image, ImageDraw

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows = []
    for row_index in range(4):
        row_dir = image_dir / str(row_index)
        row_dir.mkdir()
        row = {
            "Id": str(row_index),
            "Answer": "[1, 2, 3, 4]",
            "No_ordering": "false",
        }
        for slot in range(1, 5):
            name = f"{row_index}-{slot}.png"
            image = Image.new("L", (16, 16), color=0)
            draw = ImageDraw.Draw(image)
            draw.line(
                (0, (row_index * 4 + slot) % 16, 15, (row_index + slot * 3) % 16),
                fill=255,
                width=1,
            )
            image.save(row_dir / name)
            row[f"Input_{slot}"] = name
        rows.append(row)
    csv_path = tmp_path / "train.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(resume, "RESUME_ROOT", tmp_path / "output")

    identity = resume.training_input_identity(
        csv_path,
        image_dir,
        limit=3,
        seed=42,
        validation_fraction=0.34,
    )

    assert identity["ordered_selected_ids"] == ["1", "3", "2"]
    assert len(identity["images"]) == 12
    assert sorted(identity["split_manifest"]) == ["train_ids", "validation_ids"]
    assert Path(identity["content_addressed_manifest"]["path"]).is_file()


def test_training_command_and_provenance_use_same_resume_source(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(resume, "_source_hashes", lambda: {"src/training.py": "source-a"})
    source_attempt = tmp_path / "attempt-001"
    checkpoint = _write_valid_training_attempt(source_attempt, 5)
    training_inputs = {"tree_sha256": "dataset", "ordered_selected_ids": ["a"]}
    model = {"repository": "Qwen/Qwen3.5-27B", "tree_sha256": "model"}

    intermediate = resume.training_provenance_context(
        expected_step=5,
        limit=None,
        resume_from_checkpoint=None,
        training_inputs=training_inputs,
        model_spec=model,
    )
    continuation = resume.training_provenance_context(
        expected_step=10,
        limit=None,
        resume_from_checkpoint=checkpoint,
        training_inputs=training_inputs,
        model_spec=model,
    )
    incompatible_data = resume.training_provenance_context(
        expected_step=10,
        limit=None,
        resume_from_checkpoint=None,
        training_inputs={**training_inputs, "tree_sha256": "changed-dataset"},
        model_spec=model,
    )
    monkeypatch.setattr(
        resume, "_source_hashes", lambda: {"src/training.py": "source-b"}
    )
    incompatible_source = resume.training_provenance_context(
        expected_step=10,
        limit=None,
        resume_from_checkpoint=None,
        training_inputs=training_inputs,
        model_spec=model,
    )
    command = resume.training_command(
        stop_after_steps=10,
        save_steps=5,
        limit=None,
        resume_from_checkpoint=checkpoint,
    )

    assert (
        intermediate["resume_compatibility_key"]
        == continuation["resume_compatibility_key"]
    )
    assert (
        incompatible_data["resume_compatibility_key"]
        != continuation["resume_compatibility_key"]
    )
    assert (
        incompatible_source["resume_compatibility_key"]
        != continuation["resume_compatibility_key"]
    )
    material = continuation["parameters"]
    assert continuation["recipe_source_hashes"] == {"src/training.py": "source-a"}
    assert material["epochs"] == 6
    assert material["learning_rate"] == 1e-4
    assert material["lora_rank"] == 8
    assert material["lora_alpha"] == 32
    assert material["lora_dropout"] == 0.05
    assert material["train_vision_encoder"] is False
    assert material["balance_inputs"] is True
    assert material["clean_validation"] is True
    assert material["image_size"] == 512
    assert continuation["resume_checkpoint_identity"]["path"] == resume._project_path(
        checkpoint
    )
    index = command.index("--resume-from-checkpoint")
    assert Path(command[index + 1]).resolve() == checkpoint.resolve()


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("lora_dropout", 0.10),
        ("train_vision_encoder", True),
        ("balance_inputs", False),
        ("clean_validation", False),
    ],
)
def test_material_training_setting_change_changes_resume_compatibility_key(
    monkeypatch, field: str, changed_value: object
) -> None:
    monkeypatch.setattr(resume, "_source_hashes", lambda: {"training.py": "same"})
    training_inputs = {
        "tree_sha256": "dataset",
        "split_manifest": {"train_ids": ["a"], "validation_ids": ["b"]},
    }
    model = {
        "repository": "Qwen/Qwen3.5-27B",
        "revision": "pinned",
        "model_family": "qwen3_5",
        "verified_model_tree_sha256": "model-tree",
    }
    baseline = resume.training_provenance_context(
        expected_step=1073,
        limit=None,
        resume_from_checkpoint=None,
        training_inputs=training_inputs,
        model_spec=model,
    )
    changed = resume.training_provenance_context(
        expected_step=6438,
        limit=None,
        resume_from_checkpoint=None,
        training_inputs=training_inputs,
        model_spec=model,
        material_overrides={field: changed_value},
    )

    assert baseline["resume_compatibility_key"] != changed["resume_compatibility_key"]


def test_build_stage_provenance_round_trips_resume_compatibility(
    monkeypatch,
) -> None:
    monkeypatch.setattr(resume, "_source_hashes", lambda: {"training.py": "same"})
    context = resume.training_provenance_context(
        expected_step=1073,
        limit=None,
        resume_from_checkpoint=None,
        training_inputs={
            "tree_sha256": "dataset",
            "split_manifest": {"train_ids": ["a"], "validation_ids": ["b"]},
        },
        model_spec={
            "repository": "Qwen/Qwen3.5-27B",
            "revision": "pinned",
            "model_family": "qwen3_5",
            "verified_model_tree_sha256": "model-tree",
        },
    )
    provenance = resume.build_stage_provenance(
        "training", ["python", "-m", "snuaichal.training"], context
    )

    assert resume._valid_provenance_compatibility(
        provenance, context["resume_compatibility_key"]
    )


def _write_valid_training_attempt(attempt: Path, step: int) -> Path:
    output = attempt / "training"
    checkpoint = output / f"checkpoint-{step}"
    final = output / "final"
    checkpoint.mkdir(parents=True)
    final.mkdir(parents=True)
    (output / "training_summary.json").write_text(
        json.dumps(
            {
                "global_step": step,
                "epoch": 0.1,
                "learning_rate": 1e-4,
                "training_loss": 0.5,
                "train_runtime_seconds": 100.0,
                "train_steps_per_second": 0.02,
                "seconds_per_optimizer_step": 50.0,
                "peak_vram_bytes": 20 * 1024**3,
            }
        )
    )
    (output / "model_manifest.json").write_text(
        json.dumps(
            {
                "load_in_4bit": True,
                "trainable_parameters": 100,
                "total_parameters": 1_000,
                "model_path": "models/Qwen3.5-27B",
                "lora_rank": 8,
            }
        )
    )
    (output / "schedule.json").write_text(
        json.dumps({"stop_after_steps": step})
    )
    files = [
        final / "adapter_config.json",
        final / "adapter_model.safetensors",
        checkpoint / "adapter_config.json",
        checkpoint / "adapter_model.safetensors",
        checkpoint / "optimizer.pt",
        checkpoint / "scheduler.pt",
        checkpoint / "rng_state.pth",
        checkpoint / "training_args.bin",
    ]
    for path in files:
        path.write_bytes(b"nonempty")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step})
    )
    return checkpoint


def test_training_validator_checks_checkpoint_state_and_exact_resume(tmp_path) -> None:
    source_attempt = tmp_path / "source"
    source = _write_valid_training_attempt(source_attempt, 2)
    target_attempt = tmp_path / "target"
    target = _write_valid_training_attempt(target_attempt, 3)

    assert resume.training_validator(2)(source_attempt) == []
    assert resume.training_validator(3, resume_from_checkpoint=source)(target_attempt) == []

    (target / "trainer_state.json").write_text(json.dumps({"global_step": 2}))
    errors = resume.training_validator(3, resume_from_checkpoint=source)(target_attempt)
    assert any("trainer_state" in error for error in errors)


def _anchor_training_attempt(attempt: Path, context: dict) -> None:
    provenance = {
        "schema_version": resume.PROVENANCE_SCHEMA_VERSION,
        "stage_id": "test-training",
        "context": context,
    }
    provenance["provenance_key"] = resume.hashlib.sha256(
        resume._canonical_json(provenance).encode("utf-8")
    ).hexdigest()
    (attempt / "provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    resume._write_attempt_artifact_manifest(attempt)
    record = {
        "attempt_dir": str(attempt),
        "status": "interrupted",
        "artifact_manifest_sha256": resume.sha256_file(
            attempt / "artifact-manifest.json"
        ),
    }
    resume.REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with resume.REGISTRY_PATH.open("a", encoding="utf-8") as registry:
        registry.write(json.dumps(record) + "\n")


def _training_context(monkeypatch, *, dataset: str = "dataset") -> dict:
    monkeypatch.setattr(resume, "_source_hashes", lambda: {"training.py": "same"})
    return resume.training_provenance_context(
        expected_step=10,
        limit=None,
        resume_from_checkpoint=None,
        training_inputs={
            "tree_sha256": dataset,
            "split_manifest": {"train_ids": ["a"], "validation_ids": ["b"]},
        },
        model_spec={
            "repository": "Qwen/Qwen3.5-27B",
            "revision": "pinned",
            "model_family": "qwen3_5",
            "verified_model_tree_sha256": "model-tree",
        },
    )


def test_latest_complete_checkpoint_prefers_newest_matching_verified(
    monkeypatch, tmp_path
) -> None:
    first = _write_valid_training_attempt(tmp_path / "attempt-001", 2)
    latest = _write_valid_training_attempt(tmp_path / "attempt-002", 3)
    mismatched = _write_valid_training_attempt(tmp_path / "attempt-003", 4)
    target = _write_valid_training_attempt(tmp_path / "attempt-004", 10)
    monkeypatch.setattr(resume, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    matching_context = _training_context(monkeypatch)
    mismatched_context = _training_context(monkeypatch, dataset="other")
    _anchor_training_attempt(first.parents[1], matching_context)
    _anchor_training_attempt(latest.parents[1], matching_context)
    _anchor_training_attempt(mismatched.parents[1], mismatched_context)
    _anchor_training_attempt(target.parents[1], matching_context)
    compatibility_key = matching_context["resume_compatibility_key"]

    assert (
        resume.latest_complete_checkpoint(
            tmp_path,
            target_step=10,
            required_compatibility_key=compatibility_key,
        )
        == latest
    )
    (latest / "optimizer.pt").write_bytes(b"")
    assert (
        resume.latest_complete_checkpoint(
            tmp_path,
            target_step=10,
            required_compatibility_key=compatibility_key,
        )
        == first
    )
    (first / "training_args.bin").unlink()
    assert (
        resume.latest_complete_checkpoint(
            tmp_path,
            target_step=10,
            required_compatibility_key=compatibility_key,
        )
        is None
    )


@pytest.mark.parametrize(
    "tamper",
    ("artifact", "manifest", "missing_required", "unlisted_artifact"),
)
def test_checkpoint_selection_rejects_manifest_or_inventory_tampering(
    monkeypatch, tmp_path: Path, tamper: str
) -> None:
    stage_root = tmp_path / "stage"
    checkpoint = _write_valid_training_attempt(stage_root / "attempt-001", 5)
    monkeypatch.setattr(resume, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    context = _training_context(monkeypatch)
    _anchor_training_attempt(checkpoint.parents[1], context)

    if tamper == "artifact":
        (checkpoint / "optimizer.pt").write_bytes(b"changed")
    elif tamper == "manifest":
        manifest_path = checkpoint.parents[1] / "artifact-manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["schema_version"] = 2
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif tamper == "missing_required":
        (checkpoint / "scheduler.pt").unlink()
    else:
        (checkpoint / "unlisted-required-state.bin").write_bytes(b"unexpected")

    assert (
        resume.latest_complete_checkpoint(
            stage_root,
            target_step=10,
            required_compatibility_key=context["resume_compatibility_key"],
        )
        is None
    )


def test_prepare_full_training_resume_reuses_one_checkpoint_everywhere(
    monkeypatch, tmp_path: Path
) -> None:
    stage_root = tmp_path / "stage"
    checkpoint = _write_valid_training_attempt(stage_root / "attempt-001", 5)
    monkeypatch.setattr(resume, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    monkeypatch.setattr(resume, "_source_hashes", lambda: {"training.py": "same"})
    training_inputs = {
        "tree_sha256": "dataset",
        "split_manifest": {"train_ids": ["a"], "validation_ids": ["b"]},
    }
    model = {
        "repository": "Qwen/Qwen3.5-27B",
        "revision": "pinned",
        "model_family": "qwen3_5",
        "verified_model_tree_sha256": "model-tree",
    }
    intermediate = resume.training_provenance_context(
        expected_step=5,
        limit=None,
        resume_from_checkpoint=None,
        training_inputs=training_inputs,
        model_spec=model,
    )
    _anchor_training_attempt(checkpoint.parents[1], intermediate)
    status_updates = []
    monkeypatch.setattr(
        resume,
        "atomic_status",
        lambda **updates: status_updates.append(updates) or updates,
    )

    prepared = resume.prepare_full_training_resume(
        stage_root=stage_root,
        report_root=tmp_path / "reports",
        target_step=10,
        save_steps=5,
        training_inputs=training_inputs,
        model_spec=model,
    )

    command = prepared["command"]
    index = command.index("--resume-from-checkpoint")
    assert Path(command[index + 1]).resolve() == checkpoint.resolve()
    assert prepared["decision"]["checkpoint"] == str(checkpoint)
    assert prepared["provenance"]["resume_checkpoint_identity"]["path"] == resume._project_path(
        checkpoint
    )
    report = json.loads(Path(prepared["report"]).read_text(encoding="utf-8"))
    assert report == prepared["decision"]
    assert status_updates[-1]["full_training_resume_decision"] == prepared["decision"]
    assert status_updates[-1]["full_training_resume_report"] == prepared["report"]


def _write_valid_inference_attempt(attempt: Path, manifest: Path) -> None:
    attempt.mkdir(parents=True)
    rows = [
        {
            "Id": "a",
            "answer": [1, 2, 3, 4],
            "parse_ok": True,
            "valid_tta_views": 1,
            "aggregation_mode": "hard",
            "reference": [1, 2, 3, 4],
            "no_ordering": True,
            "views": [{}],
            "aggregations": {
                mode: {"answer": [1, 2, 3, 4], "valid_views": 1}
                for mode in resume.AGGREGATION_MODES
            },
        },
        {
            "Id": "b",
            "answer": [1, 2, 3, 4],
            "parse_ok": True,
            "valid_tta_views": 1,
            "aggregation_mode": "hard",
            "reference": [2, 1, 3, 4],
            "no_ordering": False,
            "views": [{}],
            "aggregations": {
                mode: {"answer": [1, 2, 3, 4], "valid_views": 1}
                for mode in resume.AGGREGATION_MODES
            },
        },
    ]
    with (attempt / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Id", "Answer"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"Id": row["Id"], "Answer": str(row["answer"])})
    (attempt / "audit.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    comparison = {
        mode: {"vs_hard": {"accuracy": 0.5, "exact_matches": 1}}
        for mode in resume.AGGREGATION_MODES
    }
    (attempt / "metrics.json").write_text(
        json.dumps(
            {
                "samples": 2,
                "exact_matches": 1,
                "exact_match": 0.5,
                "parse_failures": 0,
                "non_identity_exact_matches": 0,
                "non_identity_exact_match": 0.0,
                "inference_seconds_per_sample": 1.0,
                "estimated_test_seconds": 819.0,
                "peak_vram_mib": 1024.0,
                "model_precision": "nf4",
                "model_family": "qwen3_vl",
                "detected_model_family": "qwen3_vl",
                "declared_model_family": "qwen3_vl",
                "model_revision": "revision-1",
                "base_model_path": "models/Qwen3-VL-8B-Instruct",
                "adapter_path": None,
                "tta_views": 1,
                "aggregation_mode": "hard",
                "runtime_state": {
                    "quantization_applied": True,
                    "adapter_loaded": False,
                    "precision": "nf4",
                    "cuda_available": True,
                    "model_eval": True,
                    "use_cache": True,
                },
                "aggregation_comparison": comparison,
            }
        )
    )
    (attempt / "exit-code.txt").write_text("0\n")
    manifest.write_text(json.dumps({"validation_ids": ["a", "b"]}))


def test_inference_validator_reconstructs_all_gate_metrics(tmp_path) -> None:
    attempt = tmp_path / "attempt"
    manifest = tmp_path / "manifest.json"
    _write_valid_inference_attempt(attempt, manifest)
    validator = resume.inference_validator(
        2,
        adapter_loaded=False,
        validation_manifest=manifest,
        expected_tta=1,
        expected_aggregation_mode="hard",
        expected_model_path=Path("models/Qwen3-VL-8B-Instruct"),
        expected_model_family="qwen3_vl",
        expected_model_revision="revision-1",
        expected_adapter_path=None,
        expected_precision="nf4",
        uses_cuda=True,
    )
    assert validator(attempt) == []
    metrics_path = attempt / "metrics.json"
    original = json.loads(metrics_path.read_text())
    for field, wrong, message in (
        ("model_revision", "wrong", "model revision"),
        ("model_family", "qwen3_5", "model family"),
        ("adapter_path", "wrong-adapter", "adapter path"),
        ("tta_views", 4, "tta_views"),
        ("aggregation_mode", "confidence_weighted", "aggregation_mode"),
    ):
        changed = dict(original)
        changed[field] = wrong
        metrics_path.write_text(json.dumps(changed))
        assert any(message in error for error in validator(attempt))
    metrics = dict(original)
    metrics["exact_match"] = 1.0
    metrics["non_identity_exact_match"] = 1.0
    (attempt / "metrics.json").write_text(json.dumps(metrics))
    errors = validator(attempt)
    assert any("exact_match" in error for error in errors)
    assert any("non_identity" in error for error in errors)


def test_historical_inference_reuse_requires_semantics_and_exact_provenance(
    tmp_path: Path,
) -> None:
    historical = tmp_path / "historical" / "attempt-001"
    manifest = tmp_path / "manifest.json"
    _write_valid_inference_attempt(historical, manifest)
    validator = resume.inference_validator(
        2,
        adapter_loaded=False,
        validation_manifest=manifest,
        expected_tta=1,
        expected_aggregation_mode="hard",
        expected_model_path=Path("models/Qwen3-VL-8B-Instruct"),
        expected_model_family="qwen3_vl",
        expected_model_revision="revision-1",
        expected_adapter_path=None,
        expected_precision="nf4",
        uses_cuda=True,
    )
    expected_context = {"model": "exact", "validation": "exact"}
    provenance = resume.build_stage_provenance(
        "historical-4292",
        ["python", "-m", "snuaichal.inference"],
        expected_context,
    )
    (historical / "provenance.json").write_text(json.dumps(provenance))
    created = []

    selected = resume.reuse_historical_or_create(
        historical_attempt=historical,
        validator=validator,
        expected_context=expected_context,
        create_new=lambda: created.append(tmp_path / "new") or created[-1],
    )
    assert selected == historical
    assert created == []

    old_metrics = json.loads((historical / "metrics.json").read_text())
    del old_metrics["model_revision"]
    (historical / "metrics.json").write_text(json.dumps(old_metrics))
    new_attempt = tmp_path / "new" / "attempt-001"
    selected = resume.reuse_historical_or_create(
        historical_attempt=historical,
        validator=validator,
        expected_context=expected_context,
        create_new=lambda: new_attempt,
    )
    assert selected == new_attempt
    assert historical != new_attempt

    (historical / "metrics.json").write_text(json.dumps({**old_metrics, "model_revision": "revision-1"}))
    (historical / "provenance.json").write_text(json.dumps(provenance))
    selected = resume.reuse_historical_or_create(
        historical_attempt=historical,
        validator=validator,
        expected_context={"model": "different", "validation": "exact"},
        create_new=lambda: new_attempt,
    )
    assert selected == new_attempt


def test_cuda_process_query_retries_transient_nvidia_smi_failure(monkeypatch) -> None:
    calls = []

    def fake_run(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(255, "nvidia-smi")
        return subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="123, python.exe\n",
            stderr="",
        )

    monkeypatch.setattr(resume.subprocess, "run", fake_run)
    monkeypatch.setattr(resume.time, "sleep", lambda _seconds: None)

    assert resume.cuda_compute_processes() == [
        {"pid": 123, "process_name": "python.exe"}
    ]
    assert len(calls) == 2


def test_runner_lock_and_cuda_ownership_guards(monkeypatch, tmp_path) -> None:
    lock = tmp_path / "runner-lock.json"
    record = resume.acquire_runner_lock(lock)
    try:
        with pytest.raises(RuntimeError, match="runner.*active"):
            resume.acquire_runner_lock(lock)
    finally:
        resume.release_runner_lock(lock, record, state="interrupted")

    monkeypatch.setattr(
        resume,
        "cuda_compute_processes",
        lambda: [{"pid": 999, "process_name": "python.exe"}],
    )
    with pytest.raises(RuntimeError, match="CUDA"):
        resume.assert_no_foreign_cuda_processes()


def test_main_records_interrupt_between_stages(monkeypatch, tmp_path) -> None:
    events = []

    def interrupt(*, poll_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(resume, "run", interrupt)
    monkeypatch.setattr(resume, "atomic_status", lambda **payload: events.append(payload))
    lock = {"pid": 123, "started_at": "now"}
    released = []
    monkeypatch.setattr(resume, "acquire_runner_lock", lambda: lock)
    monkeypatch.setattr(
        resume,
        "release_runner_lock",
        lambda path, record, *, state: released.append((path, record, state)),
    )
    trace_path = tmp_path / "trace.json"
    trace_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        resume,
        "write_versioned_report",
        lambda *args, **kwargs: trace_path,
    )
    monkeypatch.setattr(sys, "argv", ["runner", "--poll-seconds", "1"])
    with pytest.raises(KeyboardInterrupt):
        resume.main()
    assert events[-1]["state"] == "interrupted"
    assert released == [(resume.RUNNER_LOCK_PATH, lock, "interrupted")]
