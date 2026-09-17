@echo off
if exist myenv\Scripts\activate.bat (
    call myenv\Scripts\activate.bat
) else (
    call env\Scripts\activate.bat
)
python chat.py
pause
