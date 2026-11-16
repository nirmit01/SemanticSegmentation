@echo off
REM Windows pipeline. Run from the segproject folder:  run_fast.bat
REM Safe to stop and restart: finished runs (out\TAG\val.json exists) are skipped.
setlocal
set ROOT=..\data
set EP=6
set AEP=3
set TRIALS=5

python dataset_stats.py --root %ROOT%
if errorlevel 1 goto err
if not exist results\eval\baseline_rf\metrics.json python baseline_rf.py --root %ROOT%
if errorlevel 1 goto err

REM ---- Part C: individual losses, both models (basic augmentation)
for %%M in (segformer deeplab) do (
  for %%L in (ce wce focal dice tversky lovasz) do (
    call :run %%M %%L:1 loss_%%M_%%L
    if errorlevel 1 goto err
  )
)
REM ---- manual composite
for %%M in (segformer deeplab) do (
  call :run %%M "ce:1,dice:1" comp_manual_%%M
  if errorlevel 1 goto err
)

REM ---- agent 1: loss weights (validation only)
if not exist results\agent\best_loss.txt python agent_loss.py --root %ROOT% --trials %TRIALS% --epochs %AEP% --bs 4 --workers 0
if errorlevel 1 goto err
set /p BEST=<results\agent\best_loss.txt
echo agent loss = %BEST%

for %%M in (segformer deeplab) do (
  call :run %%M "%BEST%" comp_agent_%%M
  if errorlevel 1 goto err
)

REM ---- final configuration, 3 seeds, strong augmentation
for %%M in (segformer deeplab) do (
  for %%S in (0 1 2) do (
    call :run %%M "%BEST%" final_%%M_s%%S --aug strong --seed %%S
    if errorlevel 1 goto err
  )
)

REM ---- agent 2: TTA / ensemble weight / per-class scales (validation only), then final test eval
python agent_post.py --root %ROOT% --ckpts segformer:out/final_segformer_s0/best.pt deeplab:out/final_deeplab_s0/best.pt
if errorlevel 1 goto err
python evaluate.py --root %ROOT% --ckpts segformer:out/final_segformer_s0/best.pt deeplab:out/final_deeplab_s0/best.pt --post results/agent/post_config.json --out results/eval/final_ensemble --vis 8
if errorlevel 1 goto err
python make_tables.py
echo ALL DONE - see results\tables.md
goto :eof

:run
REM %1=model %2=loss %3=tag %4..%7=extra train args
if exist out\%3\val.json (
  echo skipping %3, already done
  exit /b 0
)
python train.py --root %ROOT% --model %1 --loss %2 --epochs %EP% --bs 4 --workers 0 --tag %3 %4 %5 %6 %7
if errorlevel 1 exit /b 1
python evaluate.py --root %ROOT% --ckpts %1:out/%3/best.pt --out results/eval/%3
exit /b %errorlevel%

:err
echo FAILED - read the error above, fix it, then run run_fast.bat again (it resumes).
exit /b 1
