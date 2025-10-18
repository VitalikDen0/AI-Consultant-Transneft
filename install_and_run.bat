@echo off
chcp 65001 >nul
cls
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║          AI-Консультант Транснефть - Автоматическая установка               ║
echo ║                         1-Click Installation                                 ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.
echo [1/5] Проверка Python...

python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python не найден! Установите Python 3.10+ с https://www.python.org/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo ✅ Python %PYTHON_VERSION% найден

echo.
echo [2/5] Создание виртуального окружения...
if exist venv (
    echo ⚠️  Папка venv уже существует, пропускаем создание
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Не удалось создать venv
        pause
        exit /b 1
    )
    echo ✅ Виртуальное окружение создано
)

echo.
echo [3/5] Активация виртуального окружения...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Не удалось активировать venv
    pause
    exit /b 1
)
echo ✅ Виртуальное окружение активировано

echo.
echo [4/5] Установка зависимостей...
echo Это может занять несколько минут...
echo.

pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Ошибка установки зависимостей
    pause
    exit /b 1
)

echo.
echo ✅ Все зависимости установлены

echo.
echo [5/5] Проверка установки...
python -c "import flask, llama_cpp, cv2, mediapipe; print('✅ Основные библиотеки OK')" 2>nul
if errorlevel 1 (
    echo ⚠️  Некоторые библиотеки могут отсутствовать
)

echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                         Установка завершена!                                 ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.
echo Выберите режим запуска:
echo.
echo [1] Web-интерфейс (Flask) - Рекомендуется
echo [2] QA Benchmark (тестирование на 17 вопросах)
echo [3] Консольный режим (только LLM)
echo [0] Выход
echo.
set /p choice="Ваш выбор: "

if "%choice%"=="1" goto web
if "%choice%"=="2" goto benchmark
if "%choice%"=="3" goto console
if "%choice%"=="0" goto end

:web
echo.
echo 🌐 Запуск веб-интерфейса...
echo 📱 Откройте браузер: http://localhost:5000
echo.
python web_app.py
goto end

:benchmark
echo.
echo 📊 Запуск QA Benchmark...
echo.
python QA.py
pause
goto end

:console
echo.
echo 💻 Запуск консольного режима...
echo.
python main.py
pause
goto end

:end
echo.
echo Нажмите любую клавишу для выхода...
pause >nul
