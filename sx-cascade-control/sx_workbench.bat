@echo off
rem Start the sx-cascade-control workbench (double-click, or run from a terminal).
rem Extra arguments pass through, e.g.  sx_workbench.bat --config ndpr_plant
cd /d "%~dp0.."
pixi run workbench %*
if errorlevel 1 pause
