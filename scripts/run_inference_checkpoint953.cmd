@echo off
setlocal
cd /d "%~dp0.."
set PYTHONPATH=
if exist outputs\inference_checkpoint953_exitcode.txt del /q outputs\inference_checkpoint953_exitcode.txt
.venv\Scripts\snu-infer.exe --data-dir data --model-path models\Qwen2-VL-2B-Instruct --adapter-path outputs\qwen2-vl-lora-dhash-r16-seed42\checkpoint-953 --output outputs\submission_checkpoint953.csv --audit-log outputs\raw_predictions_checkpoint953.jsonl > outputs\inference_checkpoint953.log 2>&1
set EXIT_CODE=%ERRORLEVEL%
> outputs\inference_checkpoint953_exitcode.txt echo %EXIT_CODE%
exit /b %EXIT_CODE%
