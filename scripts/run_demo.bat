@echo off
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" src\live_demo.py --model yolo --demo-safe --weights runs\yolo_scene\best.pt %*
) else (
  python src\live_demo.py --model yolo --demo-safe --weights runs\yolo_scene\best.pt %*
)
