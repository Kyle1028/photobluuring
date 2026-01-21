@echo off
REM MySQL Database Configuration
set DATABASE_URL=mysql+pymysql://root:1028@localhost:3306/photobluuring


echo.
echo Starting Flask server
echo Browser will open automatically
echo.

start /B cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:5000"

py app.py
