@echo off
REM Complete backup script for BRAF system (Windows)
REM Updated to use correct service names (c2_server, worker_node)

setlocal enabledelayedexpansion

REM Configuration
set BACKUP_DIR=backups
set TIMESTAMP=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set BACKUP_NAME=complete_backup_%TIMESTAMP%
set BACKUP_PATH=%BACKUP_DIR%\%BACKUP_NAME%
set COMPOSE_FILE=docker-compose.prod.yml

REM Create backup directory
if not exist %BACKUP_DIR% mkdir %BACKUP_DIR%
if not exist %BACKUP_PATH% mkdir %BACKUP_PATH%

REM Initialize log file
set LOG_FILE=%BACKUP_PATH%\backup.log
echo ========================================= > %LOG_FILE%
echo BRAF Complete Backup Started at %date% %time% >> %LOG_FILE%
echo ========================================= >> %LOG_FILE%

echo [32m=========================================[0m
echo [32mBRAF Complete System Backup[0m
echo [32m=========================================[0m

REM 1. Backup PostgreSQL
echo [%time%] Backing up PostgreSQL database... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Backing up PostgreSQL database...
docker-compose -f %COMPOSE_FILE% exec -T postgres pg_dumpall -U braf_user > %BACKUP_PATH%\postgres_complete.sql 2>> %LOG_FILE%
if %errorlevel% equ 0 (
    REM Compress the SQL dump using PowerShell
    powershell -command "Compress-Archive -Path '%BACKUP_PATH%\postgres_complete.sql' -DestinationPath '%BACKUP_PATH%\postgres_complete.zip' -Force"
    del %BACKUP_PATH%\postgres_complete.sql
    echo [32m✓ PostgreSQL backup completed[0m
    echo [%time%] PostgreSQL backup completed >> %LOG_FILE%
) else (
    echo [31m✗ PostgreSQL backup failed[0m
    echo [%time%] ERROR: PostgreSQL backup failed >> %LOG_FILE%
)

REM 2. Backup Redis data
echo [%time%] Backing up Redis data... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Backing up Redis data...
docker-compose -f %COMPOSE_FILE% exec -T redis redis-cli --rdb /data/dump.rdb >> %LOG_FILE% 2>&1
if %errorlevel% equ 0 (
    docker-compose -f %COMPOSE_FILE% cp redis:/data/dump.rdb %BACKUP_PATH%\redis.rdb 2>> %LOG_FILE%
    if %errorlevel% equ 0 (
        echo [32m✓ Redis backup completed[0m
        echo [%time%] Redis backup completed >> %LOG_FILE%
    ) else (
        echo [31m✗ Failed to copy Redis dump file[0m
        echo [%time%] ERROR: Failed to copy Redis dump file >> %LOG_FILE%
    )
) else (
    echo [31m✗ Redis backup failed[0m
    echo [%time%] ERROR: Redis backup failed >> %LOG_FILE%
)

REM 3. Backup application data
echo [%time%] Backing up application data... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Backing up application data...
if exist data (
    powershell -command "Compress-Archive -Path 'data\*' -DestinationPath '%BACKUP_PATH%\app_data.zip' -Force" 2>> %LOG_FILE%
    if %errorlevel% equ 0 (
        echo [32m✓ Application data backup completed[0m
        echo [%time%] Application data backup completed >> %LOG_FILE%
    ) else (
        echo [31m✗ Application data backup failed[0m
        echo [%time%] ERROR: Application data backup failed >> %LOG_FILE%
    )
) else (
    echo [33m⚠ Application data directory not found[0m
    echo [%time%] WARNING: Application data directory not found >> %LOG_FILE%
)

REM 4. Backup recent logs (last 7 days)
echo [%time%] Backing up recent logs... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Backing up recent logs...
if exist logs (
    forfiles /p logs /m *.log /d -7 /c "cmd /c echo @path" > %TEMP%\recent_logs.txt 2>nul
    if exist %TEMP%\recent_logs.txt (
        powershell -command "Get-Content '%TEMP%\recent_logs.txt' | ForEach-Object { $_.Trim('\"') } | Compress-Archive -DestinationPath '%BACKUP_PATH%\logs_recent.zip' -Force" 2>> %LOG_FILE%
        del %TEMP%\recent_logs.txt
        echo [32m✓ Logs backup completed[0m
        echo [%time%] Logs backup completed >> %LOG_FILE%
    ) else (
        echo [33m⚠ No recent logs found[0m
        echo [%time%] WARNING: No recent logs found >> %LOG_FILE%
    )
) else (
    echo [33m⚠ Logs directory not found[0m
    echo [%time%] WARNING: Logs directory not found >> %LOG_FILE%
)

REM 5. Backup configurations
echo [%time%] Backing up system configurations... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Backing up configurations...

REM Copy configuration files
if exist %COMPOSE_FILE% copy %COMPOSE_FILE% %BACKUP_PATH%\ >nul 2>> %LOG_FILE%
if exist .env.production copy .env.production %BACKUP_PATH%\ >nul 2>> %LOG_FILE%

REM Copy configuration directories
if exist config xcopy config %BACKUP_PATH%\config\ /E /I /Q >nul 2>> %LOG_FILE%
if exist nginx xcopy nginx %BACKUP_PATH%\nginx\ /E /I /Q >nul 2>> %LOG_FILE%
if exist monitoring xcopy monitoring %BACKUP_PATH%\monitoring\ /E /I /Q >nul 2>> %LOG_FILE%
if exist grafana xcopy grafana %BACKUP_PATH%\grafana\ /E /I /Q >nul 2>> %LOG_FILE%
if exist scripts xcopy scripts %BACKUP_PATH%\scripts\ /E /I /Q >nul 2>> %LOG_FILE%

echo [32m✓ Configuration backup completed[0m
echo [%time%] Configuration backup completed >> %LOG_FILE%

REM 6. Backup certificates and uploads
echo [%time%] Backing up certificates and uploads... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Backing up certificates and uploads...

if exist certificates (
    powershell -command "Compress-Archive -Path 'certificates\*' -DestinationPath '%BACKUP_PATH%\certificates.zip' -Force" 2>> %LOG_FILE%
    echo [32m✓ Certificates backup completed[0m
)

if exist uploads (
    powershell -command "Compress-Archive -Path 'uploads\*' -DestinationPath '%BACKUP_PATH%\uploads.zip' -Force" 2>> %LOG_FILE%
    echo [32m✓ Uploads backup completed[0m
)

echo [%time%] Assets backup completed >> %LOG_FILE%

REM 7. Export Docker images
echo [%time%] Exporting Docker images... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Exporting Docker images...
docker-compose -f %COMPOSE_FILE% config --services > %TEMP%\services.txt 2>> %LOG_FILE%
if %errorlevel% equ 0 (
    set IMAGE_LIST=
    for /f %%i in (%TEMP%\services.txt) do (
        for /f %%j in ('docker-compose -f %COMPOSE_FILE% images -q %%i 2^>nul') do (
            set IMAGE_LIST=!IMAGE_LIST! %%j
        )
    )
    if defined IMAGE_LIST (
        docker save !IMAGE_LIST! -o %BACKUP_PATH%\docker_images.tar 2>> %LOG_FILE%
        if %errorlevel% equ 0 (
            powershell -command "Compress-Archive -Path '%BACKUP_PATH%\docker_images.tar' -DestinationPath '%BACKUP_PATH%\docker_images.zip' -Force"
            del %BACKUP_PATH%\docker_images.tar
            echo [32m✓ Docker images backup completed[0m
            echo [%time%] Docker images backup completed >> %LOG_FILE%
        ) else (
            echo [31m✗ Docker images export failed[0m
            echo [%time%] ERROR: Docker images export failed >> %LOG_FILE%
        )
    ) else (
        echo [33m⚠ No Docker images found[0m
        echo [%time%] WARNING: No Docker images found >> %LOG_FILE%
    )
    del %TEMP%\services.txt
) else (
    echo [31m✗ Failed to list Docker services[0m
    echo [%time%] ERROR: Failed to list Docker services >> %LOG_FILE%
)

REM 8. Create system information snapshot
echo [%time%] Creating system information snapshot... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Creating system information...
(
    echo === BRAF System Information ===
    echo Backup Date: %date% %time%
    echo System: Windows
    docker --version 2>nul
    docker-compose --version 2>nul
    echo.
    echo === Docker Containers ===
    docker-compose -f %COMPOSE_FILE% ps 2>nul
    echo.
    echo === Docker Images ===
    docker-compose -f %COMPOSE_FILE% images 2>nul
    echo.
    echo === System Resources ===
    wmic logicaldisk get size,freespace,caption 2>nul
    echo.
    echo === Network Configuration ===
    docker network ls 2>nul
) > %BACKUP_PATH%\system_info.txt 2>> %LOG_FILE%

echo [32m✓ System information created[0m
echo [%time%] System information snapshot created >> %LOG_FILE%

REM 9. Create checksums
echo [%time%] Creating checksums... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Creating checksums...
cd %BACKUP_PATH%
powershell -command "Get-ChildItem -File | ForEach-Object { $hash = Get-FileHash $_.Name -Algorithm SHA256; $hash.Algorithm + ' ' + $hash.Hash + ' ' + $hash.Path }" > checksums.sha256 2>> %LOG_FILE%
cd ..\..\
echo [32m✓ Checksums created[0m
echo [%time%] Checksums created >> %LOG_FILE%

REM 10. Create backup manifest
echo [%time%] Creating backup manifest... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Creating backup manifest...
(
echo {
echo   "backup_info": {
echo     "timestamp": "%date% %time%",
echo     "backup_type": "complete_system",
echo     "system": "BRAF",
echo     "version": "1.0.0",
echo     "backup_name": "%BACKUP_NAME%"
echo   },
echo   "components": {
echo     "database": {
echo       "type": "PostgreSQL",
echo       "file": "postgres_complete.zip",
echo       "method": "pg_dumpall",
echo       "compressed": true
echo     },
echo     "cache": {
echo       "type": "Redis",
echo       "file": "redis.rdb",
echo       "method": "redis-cli --rdb",
echo       "compressed": false
echo     }
echo   },
echo   "backup_location": "%BACKUP_PATH%",
echo   "retention_policy": "30_days"
echo }
) > %BACKUP_PATH%\backup_manifest.json

REM 11. Calculate total backup size
echo [%time%] Calculating backup size... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Calculating backup size...
for /f "tokens=3" %%a in ('dir %BACKUP_PATH% /-c ^| find "File(s)"') do set TOTAL_SIZE=%%a
echo [32m✓ Total backup size calculated[0m

REM 12. Compress entire backup
echo [%time%] Compressing complete backup... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Compressing complete backup...
powershell -command "Compress-Archive -Path '%BACKUP_PATH%\*' -DestinationPath '%BACKUP_DIR%\%BACKUP_NAME%.zip' -Force" 2>> %LOG_FILE%
if %errorlevel% equ 0 (
    REM Move log file before removing directory
    move %BACKUP_PATH%\backup.log %BACKUP_DIR%\%BACKUP_NAME%.log >nul
    rmdir /s /q %BACKUP_PATH%
    echo [32m✓ Backup compression completed[0m
    echo [%time%] Backup compression completed >> %BACKUP_DIR%\%BACKUP_NAME%.log
    set LOG_FILE=%BACKUP_DIR%\%BACKUP_NAME%.log
) else (
    echo [31m✗ Backup compression failed[0m
    echo [%time%] ERROR: Backup compression failed >> %LOG_FILE%
)

REM 13. Clean old backups (keep 30 days)
echo [%time%] Cleaning old backups... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Cleaning old backups...
forfiles /p %BACKUP_DIR% /m complete_backup_*.zip /d -30 /c "cmd /c del @path" 2>nul
forfiles /p %BACKUP_DIR% /m complete_backup_*.log /d -30 /c "cmd /c del @path" 2>nul
echo [32m✓ Old backups cleaned[0m
echo [%time%] Old backups cleaned >> %LOG_FILE%

REM 14. Verify backup integrity
echo [%time%] Verifying backup integrity... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Verifying backup integrity...
if exist %BACKUP_DIR%\%BACKUP_NAME%.zip (
    powershell -command "try { Add-Type -AssemblyName System.IO.Compression.FileSystem; [System.IO.Compression.ZipFile]::OpenRead('%BACKUP_DIR%\%BACKUP_NAME%.zip').Dispose(); exit 0 } catch { exit 1 }" 2>> %LOG_FILE%
    if %errorlevel% equ 0 (
        echo [32m✅ Backup integrity verification: PASSED[0m
        echo [%time%] Backup integrity verification: PASSED >> %LOG_FILE%
        set BACKUP_STATUS=SUCCESS
    ) else (
        echo [31m❌ Backup integrity verification: FAILED[0m
        echo [%time%] ERROR: Backup integrity verification: FAILED >> %LOG_FILE%
        set BACKUP_STATUS=FAILED
    )
) else (
    echo [31m❌ Backup file not found[0m
    echo [%time%] ERROR: Backup file not found >> %LOG_FILE%
    set BACKUP_STATUS=FAILED
)

REM Final summary
echo ========================================= >> %LOG_FILE%
echo BRAF complete backup finished at %date% %time% >> %LOG_FILE%
echo Status: %BACKUP_STATUS% >> %LOG_FILE%
echo ========================================= >> %LOG_FILE%

echo.
echo [32m=========================================[0m
echo [32mBRAF Complete Backup Summary[0m
echo [32m=========================================[0m
echo [34mStatus:[0m %BACKUP_STATUS%
echo [34mCompleted at:[0m %date% %time%
echo [34mBackup file:[0m %BACKUP_DIR%\%BACKUP_NAME%.zip
echo [34mLog file:[0m %LOG_FILE%
echo [32m=========================================[0m

if "%BACKUP_STATUS%"=="SUCCESS" (
    echo [32m✅ Complete backup completed successfully![0m
    echo Backup saved to: %BACKUP_DIR%\%BACKUP_NAME%.zip
) else (
    echo [31m❌ Backup completed with errors[0m
)

pause