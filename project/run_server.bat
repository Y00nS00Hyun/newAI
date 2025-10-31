@echo off
REM API 서버 실행 스크립트 (Windows)

echo Starting Fake News Detection API Server...
echo API Key: ULmLAYYhKeeP9J1c
echo Swagger UI: http://localhost:8000/docs
echo.

REM uvicorn으로 서버 실행
uvicorn app:app --host 0.0.0.0 --port 8000

