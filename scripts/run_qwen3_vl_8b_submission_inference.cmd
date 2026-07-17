@echo off
setlocal
cd /d "%~dp0.."
set PYTHONPATH=
set PYTHONHOME=
set PYTHONUNBUFFERED=1
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set CUDA_VISIBLE_DEVICES=0
set TOKENIZERS_PARALLELISM=false

set RUN_DIR=outputs\qwen3-vl-8b-aug
set ADAPTER=%RUN_DIR%\checkpoint-4292
set OUTPUT=outputs\submission_v5_8b_aug_checkpoint-4292_tta4.csv
set AUDIT=outputs\submission_v5_8b_aug_checkpoint-4292_tta4.jsonl
set LOG=outputs\submission_v5_8b_aug_checkpoint-4292_tta4.log
set EXIT_FILE=outputs\submission_v5_8b_aug_checkpoint-4292_tta4.exit-code.txt

if exist "%EXIT_FILE%" del /q "%EXIT_FILE%"
if exist "%OUTPUT%" del /q "%OUTPUT%"
if exist "%AUDIT%" del /q "%AUDIT%"

echo [%DATE% %TIME%] Starting snuaichallenge Qwen3-VL-8B checkpoint-4292 TTA4 inference>"%LOG%"
if not exist "%RUN_DIR%\exit-code.txt" (
  echo Training exit marker is missing>>"%LOG%"
  >"%EXIT_FILE%" echo 2
  exit /b 2
)
set /p TRAIN_EXIT=<"%RUN_DIR%\exit-code.txt"
if not "%TRAIN_EXIT%"=="0" (
  echo Training exit code is %TRAIN_EXIT%, expected 0>>"%LOG%"
  >"%EXIT_FILE%" echo 3
  exit /b 3
)
if not exist "%ADAPTER%\trainer_state.json" (
  echo Complete checkpoint-4292 is missing>>"%LOG%"
  >"%EXIT_FILE%" echo 4
  exit /b 4
)

.venv\Scripts\snu-infer.exe --data-dir data --model-path models\Qwen3-VL-8B-Instruct --adapter-path "%ADAPTER%" --load-in-4bit --image-size 512 --tta 4 --output "%OUTPUT%" --audit-log "%AUDIT%" >>"%LOG%" 2>&1
set EXIT_CODE=%ERRORLEVEL%
>"%EXIT_FILE%" echo %EXIT_CODE%
echo [%DATE% %TIME%] Inference exited with code %EXIT_CODE%>>"%LOG%"
exit /b %EXIT_CODE%
