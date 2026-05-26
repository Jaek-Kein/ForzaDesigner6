@echo off
REM Quick launcher for development. Run from the FD6\ directory.
setlocal
cd /d "%~dp0"
python -c "import logging; logging.basicConfig(level=logging.INFO, format='%%(levelname)-8s %%(name)s: %%(message)s'); import multiprocessing; multiprocessing.freeze_support(); from fd6.app import main; raise SystemExit(main())"
endlocal
