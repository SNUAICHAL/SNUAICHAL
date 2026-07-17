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
set LOG_FILE=%RUN_DIR%\training.log
set EXIT_FILE=%RUN_DIR%\exit-code.txt
set RESUME_CHECKPOINT=
set RESUME_ARG=

if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"
if exist "%EXIT_FILE%" del "%EXIT_FILE%"
for /f "usebackq delims=" %%D in (`.venv\Scripts\python.exe scripts\find_latest_checkpoint.py "%RUN_DIR%"`) do set "RESUME_CHECKPOINT=%%D"
if defined RESUME_CHECKPOINT (
  set "RESUME_ARG=--resume-from-checkpoint %RESUME_CHECKPOINT%"
  echo [%DATE% %TIME%] Resuming Qwen3-VL-8B full QLoRA training from %RESUME_CHECKPOINT%>>"%LOG_FILE%"
) else (
  echo [%DATE% %TIME%] Starting Qwen3-VL-8B full QLoRA training without a checkpoint>>"%LOG_FILE%"
)
echo Model revision: 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b>>"%LOG_FILE%"
.venv\Scripts\snu-train.exe --model-path models\Qwen3-VL-8B-Instruct --load-in-4bit --validation-fraction 0.10 --image-size 512 --epochs 6 --stop-after-steps 4292 --save-steps 250 --logging-steps 10 --batch-size 1 --gradient-accumulation-steps 8 --seed 42 --output-dir "%RUN_DIR%" %RESUME_ARG% >>"%LOG_FILE%" 2>&1
set EXIT_CODE=%ERRORLEVEL%
>"%EXIT_FILE%" echo %EXIT_CODE%
echo [%DATE% %TIME%] Training exited with code %EXIT_CODE%>>"%LOG_FILE%"
exit /b %EXIT_CODE%
