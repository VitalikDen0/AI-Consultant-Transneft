@echo off
REM Запуск импорта 3D модели в Blender
REM 
REM Этот скрипт запускает Blender и выполняет import_model.py
REM для создания transneft_character.glb из ваших OBJ + FBX файлов

echo ========================================
echo    Импорт 3D модели в Blender
echo ========================================
echo.
echo Убедитесь что у вас установлен Blender 3.0+
echo.
echo Укажите полный путь к blender.exe (например: C:\Program Files\Blender Foundation\Blender 3.6\blender.exe)
echo.

REM Попытка найти Blender автоматически
set BLENDER_PATH=

if exist "C:\Program Files\Blender Foundation\Blender 4.3\blender.exe" (
    set BLENDER_PATH=C:\Program Files\Blender Foundation\Blender 4.3\blender.exe
)

if exist "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" (
    set BLENDER_PATH=C:\Program Files\Blender Foundation\Blender 4.2\blender.exe
)

if exist "C:\Program Files\Blender Foundation\Blender 4.1\blender.exe" (
    set BLENDER_PATH=C:\Program Files\Blender Foundation\Blender 4.1\blender.exe
)

if exist "C:\Program Files\Blender Foundation\Blender 4.0\blender.exe" (
    set BLENDER_PATH=C:\Program Files\Blender Foundation\Blender 4.0\blender.exe
)

if exist "C:\Program Files\Blender Foundation\Blender 3.6\blender.exe" (
    set BLENDER_PATH=C:\Program Files\Blender Foundation\Blender 3.6\blender.exe
)

if exist "C:\Program Files\Blender Foundation\Blender 3.5\blender.exe" (
    set BLENDER_PATH=C:\Program Files\Blender Foundation\Blender 3.5\blender.exe
)

if exist "C:\Program Files\Blender Foundation\Blender 3.4\blender.exe" (
    set BLENDER_PATH=C:\Program Files\Blender Foundation\Blender 3.4\blender.exe
)

if exist "C:\Program Files\Blender Foundation\Blender 3.3\blender.exe" (
    set BLENDER_PATH=C:\Program Files\Blender Foundation\Blender 3.3\blender.exe
)

if "%BLENDER_PATH%"=="" (
    echo [ОШИБКА] Blender не найден автоматически!
    echo.
    set /p BLENDER_PATH="Введите полный путь к blender.exe: "
)

if not exist "%BLENDER_PATH%" (
    echo [ОШИБКА] Файл не найден: %BLENDER_PATH%
    echo.
    echo Установите Blender: https://www.blender.org/download/
    pause
    exit /b 1
)

echo Найден Blender: %BLENDER_PATH%
echo.
echo Запуск импорта (это может занять 1-2 минуты)...
echo.

REM Запуск Blender в фоновом режиме с выполнением скрипта
echo Команда: "%BLENDER_PATH%" --background --python "%~dp0import_model.py"
echo.
"%BLENDER_PATH%" --background --python "%~dp0import_model.py" 2>&1

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo    ИМПОРТ ЗАВЕРШЕН УСПЕШНО!
    echo ========================================
    echo.
    echo Создан файл: static\models\transneft_character.glb
    echo.
    echo Теперь запустите веб-приложение:
    echo   python web_app.py
    echo.
    echo И откройте в браузере: http://localhost:5000
    echo.
) else (
    echo.
    echo ========================================
    echo    ОШИБКА ИМПОРТА
    echo ========================================
    echo.
    echo Проверьте вывод выше для деталей ошибки
    echo.
)

pause
