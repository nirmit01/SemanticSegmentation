@echo off
REM v2 round: agent now also searches weighted CE (wce); final models trained longer.
REM Run from the segproject folder:  run_v2.bat      (safe to stop and restart; finished runs are skipped)
REM Round-1 results (run_fast.bat) are kept untouched and stay in your tables as ablations.
setlocal
set ROOT=..\data
set EP=6
set EP2=12
set A=results\agent_v2

REM ---- Agent 1 v2: search lambda for ce, wce, dice, focal, lovasz (validation only)
if not exist %A%\best_loss.txt python agent_loss.py --root %ROOT% --keys ce,wce,dice,focal,lovasz --trials 10 --epochs 4 --bs 4 --workers 0 --outdir %A% --runs_dir out_agent_v2
if errorlevel 1 goto err
set /p BEST=<%A%\best_loss.txt
echo agent v2 loss = %BEST%

REM ---- composite at the same budget as the single-loss ablation (6 epochs, basic aug)
for %%M in (segformer deeplab) do (
  call :run %%M "%BEST%" v2_comp_agent_%%M %EP%
  if errorlevel 1 goto err
)

REM ---- final models: agent loss, basic augmentation, longer training, 3 seeds
for %%M in (segformer deeplab) do (
  for %%S in (0 1 2) do (
    call :run %%M "%BEST%" v2_final_%%M_s%%S %EP2% --seed %%S
    if errorlevel 1 goto err
  )
)

REM ---- Agent 2 v2 (validation only), then final test evaluation and tables
python agent_post.py --root %ROOT% --ckpts segformer:out/v2_final_segformer_s0/best.pt deeplab:out/v2_final_deeplab_s0/best.pt --out_dir %A%
if errorlevel 1 goto err
python evaluate.py --root %ROOT% --ckpts segformer:out/v2_final_segformer_s0/best.pt deeplab:out/v2_final_deeplab_s0/best.pt --post %A%/post_config.json --out results/eval/v2_final_ensemble --vis 8
if errorlevel 1 goto err
python make_tables.py
echo ALL DONE - see results\tables.md
goto :eof

:run
REM %1=model %2=loss %3=tag %4=epochs %5..%8=extra train args
if exist out\%3\val.json (
  echo skipping %3, already done
  exit /b 0
)
python train.py --root %ROOT% --model %1 --loss %2 --epochs %4 --bs 4 --workers 0 --tag %3 %5 %6 %7 %8
if errorlevel 1 exit /b 1
python evaluate.py --root %ROOT% --ckpts %1:out/%3/best.pt --out results/eval/%3
exit /b %errorlevel%

:err
echo FAILED - read the error above, fix it, then run run_v2.bat again (it resumes).
exit /b 1
