@echo off
title Sistema de Gestion de Tickets UCompensar
color 0A

echo ========================================================
echo         Sistema de Gestion de Tickets UCompensar
echo ========================================================
echo.

:: Verificar si existe el entorno virtual
if not exist venv\ (
    echo Creando entorno virtual...
    python -m venv venv
    if errorlevel 1 (
        echo Error al crear el entorno virtual. Verifique que Python este instalado.
        pause
        exit /b 1
    )
)

:: Activar entorno virtual
echo Activando entorno virtual...
call venv\Scripts\activate
if errorlevel 1 (
    echo Error al activar el entorno virtual.
    pause
    exit /b 1
)

:: Actualizar pip y dependencias
echo Instalando dependencias...
pip install -r requirements.txt

:: Verificar y reparar el archivo CSV
echo Verificando el archivo CSV de datos...
:: No se necesita verificar el CSV, ya se maneja en clasificador.py

:: Iniciar la aplicación
echo.
echo Todo listo! Iniciando la aplicacion...
echo ========================================================
echo   La aplicacion se abrira en tu navegador automaticamente
echo ========================================================

:: Abrir el navegador y esperar 2 segundos
start "" "http://localhost:5050"
timeout /t 2 > nul

:: Iniciar el servidor
cd %~dp0
set PYTHONPATH=%CD%
python modelos_bayes\clasificador.py

pause
