@echo off
REM AlexPose Testing Framework - Windows Test Execution Script
REM Based on the Testing Strategy and Enhancement Specification

setlocal enabledelayedexpansion

REM Check if uv is available
where uv >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_CMD=uv run python
    set PYTEST_CMD=uv run pytest
    echo [INFO] Using uv for Python execution
) else (
    set PYTHON_CMD=python
    set PYTEST_CMD=pytest
    echo [WARNING] uv not found, using system python
)

REM Parse command line arguments
set COMMAND=%1
set PROFILE=dev
set PARALLEL=false
set VERBOSE=false
set NO_COV=false

if "%COMMAND%"=="" (
    echo AlexPose Testing Framework
    echo.
    echo Usage: %0 [COMMAND] [OPTIONS]
    echo.
    echo Commands:
    echo   fast          Run fast tests only ^(^< 1 second each^) - Development workflow
    echo   slow          Run slow tests ^(1-30 seconds each^)
    echo   performance   Run performance tests ^(30+ seconds^) - Benchmarking
    echo   integration   Run integration tests with external resources
    echo   property      Run property-based tests with Hypothesis
    echo   unit          Run unit tests only
    echo   e2e           Run end-to-end tests
    echo   all           Run all tests except performance
    echo   ci            Run CI/CD test suite with coverage
    echo   coverage      Generate coverage report
    echo   clean         Clean test artifacts and cache
    echo.
    echo Examples:
    echo   %0 fast                    # Quick development tests
    echo   %0 property                # Property tests
    echo   %0 all                     # All tests
    echo   %0 ci                      # Full CI test suite
    goto :eof
)

REM Execute commands
if "%COMMAND%"=="fast" (
    echo [INFO] Running fast tests ^(^< 1 second each^) - Development workflow
    %PYTEST_CMD% -v -m "fast or (not slow and not performance and not integration)"
    if !errorlevel! == 0 echo [SUCCESS] Fast tests completed
) else if "%COMMAND%"=="slow" (
    echo [INFO] Running slow tests ^(1-30 seconds each^)
    %PYTEST_CMD% -v -m slow
    if !errorlevel! == 0 echo [SUCCESS] Slow tests completed
) else if "%COMMAND%"=="performance" (
    echo [INFO] Running performance tests ^(30+ seconds^) - Benchmarking
    echo [WARNING] Performance tests may take several minutes to complete
    %PYTEST_CMD% -v -m performance --benchmark-only --no-cov
    if !errorlevel! == 0 echo [SUCCESS] Performance tests completed
) else if "%COMMAND%"=="integration" (
    echo [INFO] Running integration tests with external resources
    %PYTEST_CMD% -v -m integration
    if !errorlevel! == 0 echo [SUCCESS] Integration tests completed
) else if "%COMMAND%"=="property" (
    echo [INFO] Running property-based tests with Hypothesis
    echo [INFO] Using Hypothesis profile: %PROFILE%
    %PYTEST_CMD% -v -m property --hypothesis-profile=%PROFILE%
    if !errorlevel! == 0 echo [SUCCESS] Property-based tests completed
) else if "%COMMAND%"=="unit" (
    echo [INFO] Running unit tests
    %PYTEST_CMD% -v -m "unit or (not integration and not e2e and not performance)"
    if !errorlevel! == 0 echo [SUCCESS] Unit tests completed
) else if "%COMMAND%"=="e2e" (
    echo [INFO] Running end-to-end tests
    %PYTEST_CMD% -v -m e2e --no-cov
    if !errorlevel! == 0 echo [SUCCESS] End-to-end tests completed
) else if "%COMMAND%"=="all" (
    echo [INFO] Running all tests except performance
    %PYTEST_CMD% -v -m "not performance"
    if !errorlevel! == 0 echo [SUCCESS] All tests completed
) else if "%COMMAND%"=="ci" (
    echo [INFO] Running CI/CD test suite with comprehensive coverage
    echo [INFO] Stage 1: Fast tests
    %PYTEST_CMD% -v -m "fast or (not slow and not performance and not integration)"
    if !errorlevel! neq 0 goto :error
    
    echo [INFO] Stage 2: Unit tests
    %PYTEST_CMD% -v -m "unit or (not integration and not e2e and not performance)"
    if !errorlevel! neq 0 goto :error
    
    echo [INFO] Stage 3: Property-based tests ^(CI profile^)
    %PYTEST_CMD% -v -m property --hypothesis-profile=ci
    if !errorlevel! neq 0 goto :error
    
    echo [INFO] Stage 4: Integration tests
    %PYTEST_CMD% -v -m integration
    if !errorlevel! neq 0 goto :error
    
    echo [INFO] Stage 5: Slow tests
    %PYTEST_CMD% -v -m slow
    if !errorlevel! neq 0 goto :error
    
    echo [SUCCESS] CI test suite completed successfully
) else if "%COMMAND%"=="coverage" (
    echo [INFO] Generating comprehensive coverage report
    %PYTEST_CMD% --cov=ambient --cov=server --cov-report=html:htmlcov --cov-report=term-missing --cov-report=xml
    if !errorlevel! == 0 (
        echo [SUCCESS] Coverage report generated in htmlcov/
        echo [INFO] Opening coverage report in browser
        start htmlcov\index.html
    )
) else if "%COMMAND%"=="clean" (
    echo [INFO] Cleaning test artifacts and cache
    if exist .pytest_cache rmdir /s /q .pytest_cache
    if exist htmlcov rmdir /s /q htmlcov
    if exist .coverage del .coverage
    if exist coverage.xml del coverage.xml
    if exist .hypothesis rmdir /s /q .hypothesis
    for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
    del /s /q *.pyc >nul 2>&1
    echo [SUCCESS] Test artifacts cleaned
) else (
    echo [ERROR] Unknown command: %COMMAND%
    goto :eof
)

goto :eof

:error
echo [ERROR] Test execution failed
exit /b 1