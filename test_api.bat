@REM Test rapide du backend Flask
@REM Usage: test_api.bat <image_path>

@echo off
setlocal enabledelayedexpansion

if "%1"=="" (
    echo ❌ Usage: test_api.bat ^<image_path^>
    echo Exemple: test_api.bat uploads/raw_bill.jpg
    exit /b 1
)

if not exist "%1" (
    echo ❌ Fichier non trouvé: %1
    exit /b 1
)

echo.
echo ==================================================
echo TEST API BACKEND
echo ==================================================
echo.
echo 📷 Image: %1
echo 🔗 URL: http://localhost:5000/predict
echo.

REM Test health check d'abord
echo 🔍 Vérification du serveur...
curl -s http://localhost:5000/health | findstr "ok" >nul
if %errorlevel% neq 0 (
    echo ❌ Serveur non disponible
    echo Démarre Flask d'abord: python app.py
    exit /b 1
)
echo ✅ Serveur OK

echo.
echo 📡 Envoi de l'image...
echo.

REM Envoi de l'image
curl -X POST -F "file=@%1" http://localhost:5000/predict

echo.
echo ==================================================
echo.
