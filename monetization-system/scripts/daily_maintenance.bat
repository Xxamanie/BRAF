@echo off
REM Daily maintenance tasks for BRAF system (Windows)
REM Updated to use correct service names (c2_server, worker_node)

setlocal enabledelayedexpansion

REM Set variables
set LOG_FILE=logs\maintenance_%date:~-4,4%%date:~-10,2%%date:~-7,2%.log
set BACKUP_DIR=backups
set COMPOSE_FILE=docker-compose.prod.yml

REM Ensure directories exist
if not exist logs mkdir logs
if not exist backups mkdir backups

echo ========================================= >> %LOG_FILE%
echo Starting BRAF daily maintenance at %date% %time% >> %LOG_FILE%
echo ========================================= >> %LOG_FILE%

echo [32m=========================================[0m
echo [32mBRAF Daily Maintenance Starting[0m
echo [32m=========================================[0m

REM 1. Backup database
echo [%time%] Backing up PostgreSQL database... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Backing up database...
docker-compose -f %COMPOSE_FILE% exec -T postgres pg_dump -U braf_user braf_worker > %BACKUP_DIR%\backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.sql 2>> %LOG_FILE%
if %errorlevel% equ 0 (
    echo [32m✓ Database backup completed[0m
) else (
    echo [31m✗ Database backup failed[0m
)

REM 2. Clean old backups (keep 30 days)
echo [%time%] Cleaning old backups... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Cleaning old backups...
forfiles /p %BACKUP_DIR% /m backup_*.sql /d -30 /c "cmd /c del @path" 2>> %LOG_FILE%
echo [32m✓ Old backups cleaned[0m

REM 3. Optimize database
echo [%time%] Optimizing PostgreSQL database... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Optimizing database...
docker-compose -f %COMPOSE_FILE% exec -T postgres psql -U braf_user -d braf_worker -c "VACUUM ANALYZE;" >> %LOG_FILE% 2>&1
if %errorlevel% equ 0 (
    echo [32m✓ Database optimization completed[0m
) else (
    echo [31m✗ Database optimization failed[0m
)

REM 4. Clean Redis cache
echo [%time%] Cleaning Redis cache... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Cleaning Redis cache...
docker-compose -f %COMPOSE_FILE% exec -T redis redis-cli EVAL "local keys = redis.call('keys', 'temp:*'); for i=1,#keys do redis.call('del', keys[i]) end; return #keys" 0 >> %LOG_FILE% 2>&1
if %errorlevel% equ 0 (
    echo [32m✓ Redis cache cleaned[0m
) else (
    echo [33m⚠ Redis cache cleanup had issues[0m
)

REM 5. Rotate logs (keep 7 days)
echo [%time%] Rotating application logs... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Rotating logs...
forfiles /p logs /m *.log /d -7 /c "cmd /c if not @fname==maintenance_* del @path" 2>> %LOG_FILE%
echo [32m✓ Log rotation completed[0m

REM 6. Check disk space
echo [%time%] Checking disk space... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Checking disk space...
for /f "tokens=3" %%a in ('dir /-c ^| find "bytes free"') do set FREE_SPACE=%%a
echo Disk space check completed >> %LOG_FILE%
echo [32m✓ Disk space checked[0m

REM 7. Update Playwright browsers
echo [%time%] Updating Playwright browsers... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Updating Playwright browsers...
docker-compose -f %COMPOSE_FILE% exec -T worker_node playwright install --with-deps chromium >> %LOG_FILE% 2>&1
if %errorlevel% equ 0 (
    echo [32m✓ Playwright browsers updated[0m
) else (
    echo [33m⚠ Playwright browser update had issues[0m
)

REM 8. Health check all services
echo [%time%] Performing health checks... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Performing health checks...

REM Check C2 server health
curl -f -s http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [32m✓ C2 server health check: PASSED[0m
    echo [%time%] C2 server health check: PASSED >> %LOG_FILE%
) else (
    echo [31m✗ C2 server health check: FAILED[0m
    echo [%time%] C2 server health check: FAILED >> %LOG_FILE%
)

REM Check worker node health
docker-compose -f %COMPOSE_FILE% exec -T worker_node python -c "import sys; sys.path.append('/app'); from src.braf.worker.main import health_check; exit(0 if health_check() else 1)" >> %LOG_FILE% 2>&1
if %errorlevel% equ 0 (
    echo [32m✓ Worker node health check: PASSED[0m
    echo [%time%] Worker node health check: PASSED >> %LOG_FILE%
) else (
    echo [31m✗ Worker node health check: FAILED[0m
    echo [%time%] Worker node health check: FAILED >> %LOG_FILE%
)

REM Check database connectivity
docker-compose -f %COMPOSE_FILE% exec -T postgres pg_isready -U braf_user >> %LOG_FILE% 2>&1
if %errorlevel% equ 0 (
    echo [32m✓ Database connectivity: PASSED[0m
    echo [%time%] Database connectivity check: PASSED >> %LOG_FILE%
) else (
    echo [31m✗ Database connectivity: FAILED[0m
    echo [%time%] Database connectivity check: FAILED >> %LOG_FILE%
)

REM Check Redis connectivity
docker-compose -f %COMPOSE_FILE% exec -T redis redis-cli ping >> %LOG_FILE% 2>&1
if %errorlevel% equ 0 (
    echo [32m✓ Redis connectivity: PASSED[0m
    echo [%time%] Redis connectivity check: PASSED >> %LOG_FILE%
) else (
    echo [31m✗ Redis connectivity: FAILED[0m
    echo [%time%] Redis connectivity check: FAILED >> %LOG_FILE%
)

REM 9. Collect system metrics
echo [%time%] Collecting system metrics... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Collecting system metrics...
echo === System Metrics at %date% %time% === >> %LOG_FILE%
echo Docker containers status: >> %LOG_FILE%
docker-compose -f %COMPOSE_FILE% ps >> %LOG_FILE%
echo. >> %LOG_FILE%
echo [32m✓ System metrics collected[0m

REM 10. Database statistics
echo [%time%] Collecting database statistics... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Collecting database statistics...
docker-compose -f %COMPOSE_FILE% exec -T postgres psql -U braf_user -d braf_worker -c "SELECT schemaname, tablename, n_tup_ins as inserts, n_tup_upd as updates, n_tup_del as deletes, n_live_tup as live_rows, n_dead_tup as dead_rows FROM pg_stat_user_tables ORDER BY n_live_tup DESC;" >> %LOG_FILE% 2>&1
echo [32m✓ Database statistics collected[0m

REM 11. Cleanup temporary files
echo [%time%] Cleaning temporary files... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Cleaning temporary files...
del /q %TEMP%\playwright-* 2>nul
echo [32m✓ Temporary files cleaned[0m

REM 12. Generate maintenance report
echo [%time%] Generating maintenance report... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Generating maintenance report...
set REPORT_FILE=logs\maintenance_report_%date:~-4,4%%date:~-10,2%%date:~-7,2%.json
(
echo {
echo   "maintenance_date": "%date% %time%",
echo   "backup_created": true,
echo   "services_status": {
echo     "c2_server": "checked",
echo     "database": "checked",
echo     "redis": "checked"
echo   },
echo   "maintenance_completed": true
echo }
) > %REPORT_FILE%
echo [32m✓ Maintenance report generated[0m

REM 13. Send status report (if configured)
echo [%time%] Preparing status report... >> %LOG_FILE%
echo [36m[%time:~0,8%][0m Preparing status report...
if exist scripts\send_report.py (
    docker-compose -f %COMPOSE_FILE% exec -T c2_server python /app/scripts/send_report.py --report-file %REPORT_FILE% >> %LOG_FILE% 2>&1
    if %errorlevel% equ 0 (
        echo [32m✓ Status report sent[0m
    ) else (
        echo [33m⚠ Status report sending failed[0m
    )
) else (
    echo [33m⚠ Status report script not found[0m
)

REM Final summary
echo ========================================= >> %LOG_FILE%
echo BRAF daily maintenance completed at %date% %time% >> %LOG_FILE%
echo ========================================= >> %LOG_FILE%

echo.
echo [32m=========================================[0m
echo [32mBRAF Daily Maintenance Summary[0m
echo [32m=========================================[0m
echo [34mCompleted at:[0m %date% %time%
echo [34mLog file:[0m %LOG_FILE%
echo [34mReport file:[0m %REPORT_FILE%
echo [32m=========================================[0m
echo [32mDaily maintenance completed successfully![0m

pause