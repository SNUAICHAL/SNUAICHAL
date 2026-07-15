@echo off
setlocal
cd /d "%~dp0.."
set PYTHONUNBUFFERED=1
set RUN_DIR=outputs\qwen2-vl-lora-dhash-r16-seed42
set LOG_FILE=%RUN_DIR%\resume-training.log
set EXIT_FILE=%RUN_DIR%\resume-exit-code.txt

echo [%DATE% %TIME%] Resuming from checkpoint-200>>"%LOG_FILE%"
.venv\Scripts\snu-train.exe --output-dir "%RUN_DIR%" --epochs 1 --batch-size 2 --gradient-accumulation-steps 4 --learning-rate 2e-4 --validation-fraction 0.2 --save-steps 100 --logging-steps 5 --seed 42 --resume-from-checkpoint "%RUN_DIR%\checkpoint-200" >>"%LOG_FILE%" 2>&1
set EXIT_CODE=%ERRORLEVEL%
echo %EXIT_CODE%>"%EXIT_FILE%"
echo [%DATE% %TIME%] Training exited with code %EXIT_CODE%>>"%LOG_FILE%"
exit /b %EXIT_CODE%
