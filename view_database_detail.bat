@echo off

echo [All Records with Full Details]
echo.
"C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe" -u root -p1028 -e "USE photobluuring; SELECT * FROM media ORDER BY id DESC;"

echo.
echo.
echo [Statistics]
echo.
"C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe" -u root -p1028 -e "USE photobluuring; SELECT COUNT(*) as total, file_type, status FROM media GROUP BY file_type, status;"

echo.
echo Press any key to exit
pause >nul
