#!/usr/bin/env python3
"""
Vision Handler Module with MediaPipe Tasks API

Модуль для распознавания жестов рук с камеры используя новый MediaPipe Tasks API.
Высокая производительность и точность распознавания жестов в реальном времени.
"""

import os
import cv2
import json
import time
import threading
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Callable
from pathlib import Path

# Импорты с проверкой доступности
mediapipe = None
PIL = None

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    mediapipe = True
except ImportError:
    mediapipe = None
    mp = None
    python = None
    vision = None

try:
    from PIL import Image
    PIL = True
except ImportError:
    PIL = None

# Проверка критически важных зависимостей
MISSING_DEPS = []
if mediapipe is None:
    MISSING_DEPS.append('mediapipe')
if cv2 is None:
    MISSING_DEPS.append('opencv-python')

if MISSING_DEPS:
    print(f"⚠️  Отсутствуют vision зависимости: {', '.join(MISSING_DEPS)}")
    print("Для полной функциональности установите: pip install " + ' '.join(MISSING_DEPS))

class VisionConfig:
    """Конфигурация для vision модуля"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Импорт настроек из config.py
        try:
            import config as app_config
        except ImportError:
            raise ImportError("❌ config.py не найден! Создайте файл config.py с настройками.")
        
        # Настройки камеры
        self.camera_device_id = app_config.CAMERA_DEVICE_ID
        self.camera_fps = app_config.CAMERA_FPS
        self.camera_resolution = tuple(app_config.CAMERA_RESOLUTION)
        self.gesture_detection = app_config.GESTURE_DETECTION_ENABLED
        
        # Настройки MediaPipe Tasks
        self.min_hand_detection_confidence = app_config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE
        self.min_hand_presence_confidence = app_config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE
        self.min_tracking_confidence = app_config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE
        self.max_num_hands = app_config.MEDIAPIPE_MAX_NUM_HANDS
        
        # Настройки обработки
        self.frame_skip = app_config.FRAME_SKIP
        self.analysis_interval = app_config.GESTURE_ANALYSIS_INTERVAL
        self.gesture_confidence = app_config.GESTURE_CONFIDENCE
        self.gesture_confirmation_time = app_config.GESTURE_CONFIRMATION_TIME
        self.gesture_inactivity_timeout = app_config.GESTURE_INACTIVITY_TIMEOUT
        
        # Включение/отключение функций
        self.real_time_analysis = app_config.REAL_TIME_ANALYSIS
        self.gesture_recognition = app_config.GESTURE_RECOGNITION
        self.hand_tracking = app_config.HAND_TRACKING
        
        # Путь к модели (будет создан автоматически)
        self.model_path = self._get_model_path()
        
        print("✅ Настройки Vision загружены из config.py")
    
    def _get_model_path(self) -> str:
        """Получение пути к модели MediaPipe Tasks"""
        # Используем безопасную директорию без русских символов
        safe_model_path = r"C:\temp_mediapipe\gesture_recognizer.task"
        
        # Пытаемся найти модель в стандартных местах
        possible_paths = [
            Path(safe_model_path),
            Path(__file__).parent / "models" / "gesture_recognizer.task",
            Path.home() / ".mediapipe" / "gesture_recognizer.task",
            Path(__file__).parent / "gesture_recognizer.task"
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        # Если модель не найдена, возвращаем путь для скачивания
        return str(Path(__file__).parent / "gesture_recognizer.task")

class MediaPipeGestureRecognizer:
    """Класс для распознавания жестов с помощью MediaPipe Tasks API"""
    
    def __init__(self, config: VisionConfig):
        self.config = config
        self.recognizer = None
        
        # История жестов для анализа
        self.gesture_history = []
        self.last_detection_time = 0
        
        self._initialize_mediapipe()
    
    def _initialize_mediapipe(self):
        """Инициализация MediaPipe Tasks"""
        if not mediapipe:
            print("❌ MediaPipe не установлен")
            return
        
        try:
            # Проверяем наличие модели
            model_path = self.config.model_path
            if not Path(model_path).exists():
                print(f"⚠️  Модель не найдена: {model_path}")
                print("🔄 Попробуйте скачать модель gesture_recognizer.task")
                print("   Ссылка: https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task")
                return
            
            # Настройки для MediaPipe Tasks
            base_options = python.BaseOptions(model_asset_path=model_path)  # type: ignore
            
            # Опции для распознавания жестов
            options = vision.GestureRecognizerOptions(  # type: ignore
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,  # type: ignore  # Для начала используем режим изображения
                num_hands=self.config.max_num_hands,
                min_hand_detection_confidence=self.config.min_hand_detection_confidence,
                min_hand_presence_confidence=self.config.min_hand_presence_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence
            )
            
            # Создание распознавателя
            self.recognizer = vision.GestureRecognizer.create_from_options(options)  # type: ignore
            
            print("✅ MediaPipe Tasks инициализирован успешно")
            
        except Exception as e:
            print(f"❌ Ошибка инициализации MediaPipe Tasks: {e}")
    
    def detect_hand_gestures(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Детекция жестов рук в кадре используя MediaPipe Tasks
        """
        current_time = time.time()
        if current_time - self.last_detection_time < self.config.analysis_interval:
            return []
        
        self.last_detection_time = current_time
        
        if self.recognizer is None:
            return []
        
        gestures = []
        
        try:
            # Конвертация OpenCV изображения в MediaPipe Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not mp:
                return []
                
            # Преобразование в MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Распознавание жестов
            result = self.recognizer.recognize(mp_image)
            
            # Обработка результатов
            if result.gestures:
                for i, gesture_list in enumerate(result.gestures):
                    if gesture_list:  # Если есть распознанные жесты
                        # Информация о руке (правая/левая)
                        handedness = result.handedness[i][0] if result.handedness and len(result.handedness) > i else None
                        hand_label = handedness.category_name.lower() if handedness else "unknown"
                        hand_confidence = handedness.score if handedness else 0.0
                        
                        # Информация о жесте
                        gesture = gesture_list[0]  # Берем жест с максимальной уверенностью
                        
                        # Ориентиры руки
                        landmarks = []
                        if result.hand_landmarks and len(result.hand_landmarks) > i:
                            for landmark in result.hand_landmarks[i]:
                                landmarks.append([landmark.x, landmark.y, landmark.z])
                        
                        if gesture.score > self.config.gesture_confidence:
                            gesture_data = {
                                'type': gesture.category_name.lower(),
                                'hand': hand_label,
                                'confidence': gesture.score,
                                'hand_confidence': hand_confidence,
                                'landmarks': landmarks,
                                'timestamp': current_time,
                                'description': f"{gesture.category_name} - {hand_label} рука"
                            }
                            gestures.append(gesture_data)
            
        except Exception as e:
            print(f"❌ Ошибка детекции жестов: {e}")
        
        return gestures
    
    def get_gesture_description(self, gesture: Dict[str, Any]) -> str:
        """
        Получение подробного описания жеста
        """
        gesture_descriptions = {
            "thumb_up": "👍 Большой палец вверх",
            "thumb_down": "👎 Большой палец вниз",
            "pointing_up": "☝️ Указательный палец вверх", 
            "open_palm": "✋ Открытая ладонь",
            "closed_fist": "✊ Кулак",
            "victory": "✌️ Знак победы",
            "iloveyou": "🤟 Знак 'Я люблю тебя'",
            "none": "❓ Неизвестный жест",
            "unknown": "❓ Неизвестный жест"
        }
        
        gesture_type = gesture.get('type', 'unknown')
        base_description = gesture_descriptions.get(gesture_type, f"Жест: {gesture_type}")
        hand_info = f" ({gesture['hand']} рука)"
        confidence_info = f" [Уверенность: {gesture['confidence']:.2f}]"
        
        return base_description + hand_info + confidence_info
    
    def close(self):
        """Закрытие ресурсов MediaPipe"""
        if self.recognizer:
            self.recognizer.close()

class CameraHandler:
    """Класс для работы с камерой"""
    
    def __init__(self, config: VisionConfig):
        self.config = config
        self.camera = None
        self.is_recording = False
        self.frame_callback = None
        self.current_frame = None
        self.capture_thread = None  # Добавлено: ссылка на поток захвата
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Инициализация камеры"""
        try:
            self.camera = cv2.VideoCapture(self.config.camera_device_id)
            
            # Настройка параметров камеры
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_resolution[1])
            self.camera.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
            
            # Проверка успешного открытия
            if not self.camera.isOpened():
                raise Exception("Не удалось открыть камеру")
            
            print(f"📹 Камера инициализирована: {self.config.camera_resolution} @ {self.config.camera_fps}fps")
            
        except Exception as e:
            print(f"❌ Ошибка инициализации камеры: {e}")
            self.camera = None
    
    def start_capture(self, frame_callback: Optional[Callable] = None):
        """Запуск захвата видео"""
        # Реинициализируем камеру если она была освобождена
        if self.camera is None:
            print("🔄 Реинициализация камеры...")
            self._initialize_camera()
            if self.camera is None:
                print("❌ Не удалось инициализировать камеру")
                return False
        
        if self.is_recording:
            print("⚠️ Захват видео уже запущен")
            return True
        
        self.frame_callback = frame_callback
        self.is_recording = True
        
        # Запуск в отдельном потоке
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        print("🎥 Захват видео запущен")
        return True
    
    def stop_capture(self):
        """Остановка захвата видео и освобождение камеры"""
        self.is_recording = False
        
        # Ждём завершения потока захвата
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        self.capture_thread = None  # Обнуляем ссылку на поток
        
        # Освобождаем камеру
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            print("📹 Камера освобождена")
        
        self.current_frame = None
        print("⏹️  Захват видео остановлен")
    
    def _capture_loop(self):
        """Основной цикл захвата кадров"""
        frame_count = 0
        
        while self.is_recording and self.camera is not None:
            ret, frame = self.camera.read()
            
            if not ret:
                print("❌ Ошибка чтения кадра с камеры")
                break
            
            # Пропуск кадров для экономии ресурсов
            frame_count += 1
            if frame_count % self.config.frame_skip != 0:
                continue
            
            self.current_frame = frame
            
            # Вызов callback функции если задана
            if self.frame_callback:
                try:
                    self.frame_callback(frame)
                except Exception as e:
                    print(f"❌ Ошибка в callback: {e}")
            
            # Ограничение FPS
            time.sleep(1.0 / self.config.camera_fps)
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Получение текущего кадра"""
        return self.current_frame
    
    def release(self):
        """Освобождение ресурсов камеры"""
        self.stop_capture()
        if self.camera:
            self.camera.release()
            cv2.destroyAllWindows()

class VisionHandler:
    """Основной класс для работы с vision модулем"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = VisionConfig(config_path)
        self.camera_handler = CameraHandler(self.config)
        self.gesture_recognizer = MediaPipeGestureRecognizer(self.config)
        
        # Состояние
        self.is_vision_active = False
        self.is_paused = False  # Флаг паузы (не останавливаем камеру, просто не обрабатываем)
        self.last_analysis_time = 0
        self.analysis_results = []
        self.current_gestures = []
        
        # Логика подтверждения жестов
        self.gesture_buffer = []  # Буфер жестов за последние N секунд
        self.last_gesture_time = time.time()  # Время последнего распознанного жеста
        self.last_confirmed_gesture = None  # Последний подтверждённый жест
        self.gesture_callback = None  # Callback для отправки жеста
        
        print("👁️  Vision модуль инициализирован")
        self.print_status()
    
    def print_status(self):
        """Вывод статуса vision модуля"""
        print(f"📹 Камера: {'✅' if self.camera_handler.camera else '❌'}")
        print(f"🤖 MediaPipe Tasks: {'✅' if self.gesture_recognizer.recognizer else '❌'}")
        print(f"👋 Детекция жестов: {'✅' if self.config.gesture_recognition else '❌'}")
        
        if not self.gesture_recognizer.recognizer:
            print(f"📁 Путь к модели: {self.config.model_path}")
            print("💡 Совет: Скачайте gesture_recognizer.task модель для работы")
    
    def set_gesture_callback(self, callback):
        """Установка callback функции для подтверждённых жестов"""
        self.gesture_callback = callback
    
    def _check_gesture_confirmation(self):
        """Проверка подтверждения жеста на основе буфера"""
        if not self.gesture_buffer:
            return
        
        # Подсчёт жестов по типу в буфере
        gesture_counts = {}
        for gesture in self.gesture_buffer:
            gesture_type = gesture['type']  # Исправлено: 'type' вместо 'gesture'
            if gesture_type not in gesture_counts:
                gesture_counts[gesture_type] = 0
            gesture_counts[gesture_type] += 1
        
        # Находим самый частый жест
        if gesture_counts:
            most_common_gesture = max(gesture_counts, key=gesture_counts.get)
            count = gesture_counts[most_common_gesture]
            
            # Если жест появился достаточно раз (минимум 3 раза за 1.5 секунды при 16fps = ~24 кадра)
            # И это не тот же жест что недавно подтверждали (чтобы избежать дубликатов)
            if count >= 3 and most_common_gesture != self.last_confirmed_gesture:
                # Маппинг жестов на текст
                gesture_text_map = {
                    'open_palm': 'Привет',
                    'victory': 'Победа',
                    'thumb_up': 'Супер',
                    'pointing_up': 'Внимание'
                }
                
                if most_common_gesture in gesture_text_map:
                    confirmed_text = gesture_text_map[most_common_gesture]
                    print(f"✅ Подтверждён жест: {most_common_gesture} → '{confirmed_text}'")
                    
                    # Обновляем последний подтверждённый жест
                    self.last_confirmed_gesture = most_common_gesture
                    
                    # Вызываем callback если установлен
                    if self.gesture_callback:
                        self.gesture_callback(confirmed_text)
                    
                    # Очищаем буфер после подтверждения
                    self.gesture_buffer = []
    
    def start_real_time_analysis(self, gesture_callback=None):
        """Запуск анализа в реальном времени"""
        try:
            if not self.config.real_time_analysis:
                print("❌ Анализ в реальном времени отключен в конфиге")
                return False
            
            if self.camera_handler.camera is None:
                print("❌ Камера недоступна")
                return False
            
            if self.gesture_recognizer.recognizer is None:
                print("⚠️ MediaPipe Tasks недоступен - будет работать только камера без анализа жестов")
                # Не возвращаем False - камера может работать и без жестов
            
            # Устанавливаем callback если передан
            if gesture_callback:
                self.gesture_callback = gesture_callback
            
            # Сброс состояния
            self.gesture_buffer = []
            self.last_gesture_time = time.time()
            self.last_confirmed_gesture = None
            
            self.is_vision_active = True
            success = self.camera_handler.start_capture(self._frame_analysis_callback)
            
            if success:
                print("🔍 Анализ жестов в реальном времени запущен")
            else:
                print("❌ Не удалось запустить захват с камеры")
            return success
        except Exception as e:
            print(f"❌ Ошибка запуска анализа: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def pause_analysis(self):
        """Приостановка анализа (камера продолжает работать, но жесты не обрабатываются)"""
        self.is_paused = True
        print("⏸️ Анализ жестов приостановлен (камера активна)")
    
    def resume_analysis(self):
        """Возобновление анализа жестов"""
        if self.is_vision_active:
            self.is_paused = False
            self.last_gesture_time = time.time()  # Сброс таймера
            self.gesture_buffer = []  # Очистка буфера
            print("▶️ Анализ жестов возобновлён")
        else:
            print("⚠️ Анализ не был запущен, используйте start_real_time_analysis()")
    
    def stop_real_time_analysis(self):
        """Остановка анализа в реальном времени"""
        try:
            self.is_vision_active = False
            self.is_paused = False
            self.camera_handler.stop_capture()
            print("⏹️  Анализ жестов остановлен")
        except Exception as e:
            print(f"❌ Ошибка остановки анализа: {e}")
            import traceback
            traceback.print_exc()
    
    def _frame_analysis_callback(self, frame: np.ndarray):
        """Callback для анализа кадров"""
        current_time = time.time()
        
        try:
            # Пропускаем обработку если анализ на паузе
            if self.is_paused:
                return
            
            # Проверка таймаута бездействия
            if current_time - self.last_gesture_time > self.config.gesture_inactivity_timeout:
                print(f"⏰ Таймаут бездействия ({self.config.gesture_inactivity_timeout} сек) - останавливаем распознавание жестов")
                self.stop_real_time_analysis()
                return
            
            # Детекция жестов
            gestures = []
            if self.config.gesture_recognition and self.gesture_recognizer.recognizer:
                gestures = self.gesture_recognizer.detect_hand_gestures(frame)
            
            # Обновление текущих жестов
            self.current_gestures = gestures
            
            # Обработка жестов
            if gestures:
                # Обновляем время последнего жеста
                self.last_gesture_time = current_time
                
                # Добавляем жесты в буфер
                for gesture in gestures:
                    gesture['detected_at'] = current_time
                    self.gesture_buffer.append(gesture)
                    
                    description = self.gesture_recognizer.get_gesture_description(gesture)
                    print(f"👋 {description}")
                
                # Очищаем старые жесты из буфера (старше чем confirmation_time)
                self.gesture_buffer = [
                    g for g in self.gesture_buffer 
                    if current_time - g['detected_at'] <= self.config.gesture_confirmation_time
                ]
                
                # Проверяем подтверждение жеста
                self._check_gesture_confirmation()
                
                # Сохранение результатов
                result = {
                    'timestamp': current_time,
                    'gestures': gestures,
                    'frame_shape': frame.shape
                }
                
                self.analysis_results.append(result)
                
                # Ограничение размера истории
                if len(self.analysis_results) > 20:
                    self.analysis_results.pop(0)
            
        except Exception as e:
            print(f"❌ Ошибка анализа кадра: {e}")
            import traceback
            traceback.print_exc()
    
    def get_current_gestures(self) -> List[Dict[str, Any]]:
        """Получение текущих жестов"""
        return self.current_gestures
    
    def get_recent_analysis(self, count: int = 5) -> List[Dict[str, Any]]:
        """Получение последних результатов анализа"""
        return self.analysis_results[-count:] if self.analysis_results else []
    
    def analyze_single_frame(self, prompt: str = "Анализ жестов") -> str:
        """Анализ одиночного кадра"""
        frame = self.camera_handler.get_current_frame()
        if frame is None:
            return "Кадр недоступен"
        
        if self.gesture_recognizer.recognizer is None:
            return "MediaPipe Tasks недоступен"
        
        gestures = self.gesture_recognizer.detect_hand_gestures(frame)
        
        if not gestures:
            return "Жесты не обнаружены"
        
        descriptions = []
        for gesture in gestures:
            desc = self.gesture_recognizer.get_gesture_description(gesture)
            descriptions.append(desc)
        
        return f"Обнаружены жесты: {', '.join(descriptions)}"
    
    def is_camera_available(self) -> bool:
        """Проверка доступности камеры"""
        return self.camera_handler.camera is not None
    
    def is_mediapipe_available(self) -> bool:
        """Проверка доступности MediaPipe Tasks"""
        return self.gesture_recognizer.recognizer is not None
    
    def list_cameras(self) -> List[int]:
        """Получение списка доступных камер"""
        available_cameras = []
        print("📹 Поиск доступных камер...")
        
        # Проверяем камеры от 0 до 10
        for i in range(11):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Проверяем, можем ли получить кадр
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        available_cameras.append(i)
                        
                        # Получаем информацию о камере
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        print(f"   ✓ Камера {i}: {width}x{height} @ {fps:.1f}fps")
                    
                cap.release()
            except Exception:
                continue
        
        if available_cameras:
            print(f"📹 Найдено камер: {len(available_cameras)} - {available_cameras}")
        else:
            print("❌ Камеры не найдены")
            
        return available_cameras
    
    def switch_camera(self, camera_id: int) -> bool:
        """Переключение на другую камеру"""
        available_cameras = self.list_cameras()
        
        if camera_id not in available_cameras:
            print(f"❌ Камера {camera_id} недоступна. Доступные: {available_cameras}")
            return False
        
        # Останавливаем текущий анализ
        was_running = self.is_vision_active
        if was_running:
            self.stop_real_time_analysis()
        
        # Обновляем конфигурацию
        self.config.camera_device_id = camera_id
        
        # Переинициализируем камеру
        self.camera_handler.release()
        self.camera_handler = CameraHandler(self.config)
        
        # Возобновляем анализ, если он был запущен
        if was_running:
            success = self.start_real_time_analysis()
            if success:
                print(f"📹 Переключились на камеру {camera_id}")
            return success
        else:
            print(f"📹 Камера переключена на {camera_id}")
            return True
    
    def get_current_camera(self) -> int:
        """Получение ID текущей камеры"""
        return self.config.camera_device_id
    
    def refresh_cameras(self) -> List[int]:
        """Обновление списка доступных камер"""
        return self.list_cameras()
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Получение текущего кадра с камеры"""
        return self.camera_handler.get_current_frame()
    
    def cleanup(self):
        """Очистка ресурсов"""
        self.stop_real_time_analysis()
        self.camera_handler.release()
        self.gesture_recognizer.close()

# Пример использования
if __name__ == "__main__":
    print("👁️  Тестирование vision модуля с MediaPipe Tasks...")
    
    vision_handler = VisionHandler()
    
    if vision_handler.is_camera_available():
        print("\n📹 Тест камеры и распознавания жестов:")
        
        if vision_handler.is_mediapipe_available():
            vision_handler.start_real_time_analysis()
            
            try:
                # Тест в течение 10 секунд
                print("Покажите различные жесты перед камерой...")
                time.sleep(10)
                
                # Анализ текущего кадра
                analysis = vision_handler.analyze_single_frame()
                print(f"Анализ кадра: {analysis}")
                
                # Показать последние результаты
                recent = vision_handler.get_recent_analysis()
                print(f"Последние анализы: {len(recent)}")
                
                # Текущие жесты
                current = vision_handler.get_current_gestures()
                print(f"Текущие жесты: {len(current)}")
                
            finally:
                vision_handler.cleanup()
        else:
            print("❌ MediaPipe Tasks недоступен - требуется модель gesture_recognizer.task")
    
    print("\n✅ Тестирование завершено!")