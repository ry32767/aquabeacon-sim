@echo off
REM ============================================================================
REM  run_all.bat - AquaBeacon: run every demonstration scenario in one shot.
REM
REM  Runs all scripts/run_*.py sequentially. Each scenario writes its figures,
REM  JSON/CSV and an auto-generated README under results/<scenario>/.
REM  Failures do not stop the batch; a summary is printed at the end.
REM
REM  Usage (double-click, or from a terminal at the repo root):
REM      run_all.bat
REM  Optionally pick a Python:  set PYTHON=py -3  &  run_all.bat
REM
REM  NOTE: wrapper messages are ASCII on purpose (cmd.exe reads .bat in the
REM  console code page). The Python scripts print their own Japanese output.
REM ============================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

if "%PYTHON%"=="" set "PYTHON=python"

REM Order: quick scenarios first, heavier ones (spec, visualize) last.
set "SCENARIOS=run_minimum run_sensitivity run_mapping run_robust run_deepwater run_depth run_no_optical run_sbl run_opmap run_switch run_attitude run_spec run_visualize"

set /a TOTAL=0
set /a FAILED=0
set "FAILLIST="

echo ============================================================
echo  AquaBeacon - run all scenarios
echo  Python: %PYTHON%
echo ============================================================

for %%S in (%SCENARIOS%) do (
    set /a TOTAL+=1
    echo.
    echo ------------------------------------------------------------
    echo  [!TOTAL!] %%S
    echo ------------------------------------------------------------
    "%PYTHON%" "scripts\%%S.py"
    if errorlevel 1 (
        set /a FAILED+=1
        set "FAILLIST=!FAILLIST! %%S"
        echo  [NG] %%S failed
    ) else (
        echo  [OK] %%S
    )
)

set /a PASSED=TOTAL-FAILED
echo.
echo ============================================================
echo  Done: !PASSED!/!TOTAL! scenarios OK, !FAILED! failed
if not "!FAILLIST!"=="" echo  Failed:!FAILLIST!
echo  See results\ (index: results\README.md) for outputs.
echo ============================================================
endlocal
pause
