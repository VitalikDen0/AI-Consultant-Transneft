#!/usr/bin/env python3
"""
Audio Handler Module

Модуль для распознавания речи и генерации аудио ответов.
Поддерживает как API решения, так и локальные модели.
"""

import os
import io
import wave
import json
import tempfile
import threading
from typing import Optional, Dict, Any, Callable, Union
from pathlib import Path

# Импорт настроек из config.py
try:
    import config as app_config
except ImportError:
    raise ImportError("❌ config.py не найден! Создайте файл config.py с настройками.")

# Импорты с проверкой доступности
pyaudio = None
sr = None
pyttsx3 = None
requests = None
AudioSegment = None
play = None
np = None

try:
    import pyaudio
except ImportError:
    pyaudio = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    import requests
except ImportError:
    requests = None

try:
    from pydub import AudioSegment
    from pydub.playback import play
except ImportError:
    AudioSegment = None
    play = None

try:
    import numpy as np
except ImportError:
    np = None

# Проверка критически важных зависимостей
MISSING_DEPS = []
if sr is None:
    MISSING_DEPS.append('speechrecognition')
if pyttsx3 is None:
    MISSING_DEPS.append('pyttsx3')
if pyaudio is None:
    MISSING_DEPS.append('pyaudio')
if requests is None:
    MISSING_DEPS.append('requests')
if np is None:
    MISSING_DEPS.append('numpy')

if MISSING_DEPS:
    print(f"⚠️  Отсутствуют аудио зависимости: {', '.join(MISSING_DEPS)}")
    print("Для полной функциональности установите: pip install " + ' '.join(MISSING_DEPS))

class AudioConfig:
    """Конфигурация для аудио модуля"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Загружаем всё из config.py
        self._load_from_python_config()
    
    def _load_from_python_config(self):
        """Загрузка настроек из config.py"""
        # Настройки распознавания речи
        self.stt_engine = app_config.STT_ENGINE
        self.stt_language = app_config.STT_LANGUAGE
        self.stt_enabled = app_config.STT_ENABLED
        self.whisper_api_key = app_config.WHISPER_API_KEY
        self.whisper_api_url = app_config.WHISPER_API_URL
        
        # Настройки синтеза речи
        self.tts_engine = app_config.TTS_ENGINE
        self.tts_voice = app_config.TTS_VOICE
        self.tts_rate = app_config.TTS_RATE
        self.tts_volume = app_config.TTS_VOLUME
        self.tts_enabled = app_config.TTS_ENABLED
        self.auto_play = app_config.TTS_AUTO_PLAY
        
        # API ключи
        self.elevenlabs_api_key = app_config.ELEVENLABS_API_KEY
        self.elevenlabs_voice_id = app_config.ELEVENLABS_VOICE_ID
        
        # Настройки записи
        self.sample_rate = app_config.AUDIO_SAMPLE_RATE
        self.chunk_size = app_config.AUDIO_CHUNK_SIZE
        self.channels = 1  # Всегда моно
        self.record_timeout = 10  # Таймаут записи
        
        print("✅ Настройки загружены из config.py")

class SpeechRecognizer:
    """Класс для распознавания речи"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.recognizer = None
        self.microphone = None
        self._initialize_recognizer()
    
    def _initialize_recognizer(self):
        """Инициализация распознавателя речи"""
        if not self.config.stt_enabled or sr is None:
            if sr is None:
                print("⚠️  speechrecognition не установлен")
            self.config.stt_enabled = False
            return
            
        try:
            self.recognizer = sr.Recognizer()
            if pyaudio is not None:
                self.microphone = sr.Microphone(sample_rate=self.config.sample_rate)
                
                # Калибровка шума
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
            else:
                print("⚠️  pyaudio не установлен, микрофон недоступен")
                self.config.stt_enabled = False
                return
                
            print("🎤 Распознавание речи инициализировано")
        except Exception as e:
            print(f"⚠️  Ошибка инициализации распознавания речи: {e}")
            self.config.stt_enabled = False
    
    def listen_for_speech(self, timeout: Optional[int] = None) -> Optional[str]:
        """
        Слушает речь и возвращает распознанный текст
        """
        if not self.config.stt_enabled or not self.recognizer or not self.microphone:
            return None
        
        timeout = timeout or self.config.record_timeout
        
        try:
            print("🎤 Говорите... (нажмите Ctrl+C для остановки)")
            
            with self.microphone as source:
                # Запись аудио
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=30)
            
            print("🔄 Распознавание речи...")
            
            # Выбор движка распознавания
            if self.config.stt_engine == 'google' and self.recognizer:
                # type: ignore - Проверяем что recognizer не None
                text = self.recognizer.recognize_google(audio, language=self.config.stt_language)  # type: ignore
            elif self.config.stt_engine == 'whisper_api':
                text = self._recognize_with_whisper_api(audio)
            elif self.config.stt_engine == 'local_whisper':
                text = self._recognize_with_local_whisper(audio)
            else:
                if self.recognizer:
                    # type: ignore - Проверяем что recognizer не None
                    text = self.recognizer.recognize_google(audio, language=self.config.stt_language)  # type: ignore
                else:
                    raise RuntimeError("Google Speech Recognition недоступен")
            
            return text
            
        except Exception as e:
            # Обработка всех исключений sr, если модуль доступен
            if sr is not None:
                if isinstance(e, sr.WaitTimeoutError):
                    print("⏰ Таймаут записи")
                    return None
                elif isinstance(e, sr.UnknownValueError):
                    print("❓ Не удалось распознать речь")
                    return None
                elif isinstance(e, sr.RequestError):
                    print(f"❌ Ошибка сервиса распознавания: {e}")
                    return None
            
            print(f"❌ Ошибка распознавания: {e}")
            return None
    
    def _recognize_with_whisper_api(self, audio) -> str:
        """Распознавание через Whisper API"""
        if not self.config.whisper_api_key:
            raise Exception("Whisper API ключ не настроен")
        
        if requests is None:
            raise Exception("requests библиотека не установлена")
        
        # Конвертация в WAV формат
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.config.sample_rate)
                wav_file.writeframes(audio.get_wav_data())
            
            # Отправка на Whisper API
            with open(tmp_file.name, 'rb') as audio_file:
                headers = {'Authorization': f'Bearer {self.config.whisper_api_key}'}
                files = {'file': audio_file}
                data = {'model': 'whisper-1', 'language': 'ru'}
                
                response = requests.post(self.config.whisper_api_url, headers=headers, files=files, data=data)
                response.raise_for_status()
                
                result = response.json()
                text = result.get('text', '')
            
            # Удаление временного файла
            try:
                os.unlink(tmp_file.name)
            except:
                pass
            
            return text
    
    def _recognize_with_local_whisper(self, audio) -> str:
        """Заглушка для локального Whisper"""
        # TODO: Интеграция с локальной моделью Whisper
        print("🔧 Локальный Whisper пока не реализован, используется Google")
        if self.recognizer is not None:
            if self.recognizer:
                # type: ignore - Проверяем что recognizer не None
                return self.recognizer.recognize_google(audio, language=self.config.stt_language)  # type: ignore
            else:
                raise RuntimeError("Google Speech Recognition недоступен")
        else:
            raise Exception("Распознаватель речи не инициализирован")

class TextToSpeech:
    """Класс для синтеза речи"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.engine = None
        self._initialize_tts()
    
    def _initialize_tts(self):
        """Инициализация TTS движка"""
        if not self.config.tts_enabled or pyttsx3 is None:
            if pyttsx3 is None:
                print("⚠️  pyttsx3 не установлен")
            self.config.tts_enabled = False
            return
            
        try:
            if self.config.tts_engine == 'pyttsx3':
                self.engine = pyttsx3.init()
                
                # Настройка голоса (выбираем мужской русский)
                voices = self.engine.getProperty('voices')
                male_voice_found = False
                
                if voices:  # Проверка что voices не None
                    try:
                        # Сначала ищем мужской русский голос
                        for voice in voices:  # type: ignore
                            if hasattr(voice, 'name') and hasattr(voice, 'id'):
                                name_lower = voice.name.lower()
                                id_lower = voice.id.lower()
                                
                                # Ищем мужской русский голос (приоритет)
                                if ('russian' in name_lower or 'ru' in id_lower) and \
                                   ('male' in name_lower or 'ivan' in name_lower or 'dmitry' in name_lower):
                                    self.engine.setProperty('voice', voice.id)
                                    male_voice_found = True
                                    print(f"🎙️ Выбран мужской голос: {voice.name}")
                                    break
                        
                        # Если не нашли мужской, берём любой русский
                        if not male_voice_found:
                            for voice in voices:  # type: ignore
                                if hasattr(voice, 'name') and hasattr(voice, 'id'):
                                    if 'russian' in voice.name.lower() or 'ru' in voice.id.lower():
                                        self.engine.setProperty('voice', voice.id)
                                        print(f"🎙️ Выбран голос: {voice.name}")
                                        break
                    except (TypeError, AttributeError):
                        pass  # Игнорируем ошибки с типами
                
                # Настройка скорости (увеличиваем на 20% для ускорения)
                faster_rate = int(self.config.tts_rate * 1.2)
                self.engine.setProperty('rate', faster_rate)
                self.engine.setProperty('volume', self.config.tts_volume)
                
                print(f"⚡ Скорость речи: {faster_rate} слов/мин")
                
            print("🔊 Синтез речи инициализирован")
        except Exception as e:
            print(f"⚠️  Ошибка инициализации TTS: {e}")
            self.config.tts_enabled = False
    
    def speak_text(self, text: str, play_async: bool = True) -> Optional[str]:
        """
        Озвучивает текст
        """
        if not self.config.tts_enabled or not text.strip():
            return None
        
        try:
            if self.config.tts_engine == 'pyttsx3':
                return self._speak_with_pyttsx3(text, play_async)
            elif self.config.tts_engine == 'elevenlabs':
                return self._speak_with_elevenlabs(text, play_async)
            elif self.config.tts_engine == 'local_tts':
                return self._speak_with_local_tts(text, play_async)
            else:
                return self._speak_with_pyttsx3(text, play_async)
                
        except Exception as e:
            print(f"❌ Ошибка синтеза речи: {e}")
            return None
    
    def _speak_with_pyttsx3(self, text: str, play_async: bool) -> Optional[str]:
        """Озвучивание через pyttsx3"""
        if not self.engine or not hasattr(self.engine, 'say'):
            return None
        
        def speak():
            try:
                if self.engine and hasattr(self.engine, 'say') and hasattr(self.engine, 'runAndWait'):
                    self.engine.say(text)
                    self.engine.runAndWait()
                else:
                    print("❌ TTS Engine недоступен")
            except Exception as e:
                print(f"❌ Ошибка озвучивания: {e}")
        
        if play_async:
            thread = threading.Thread(target=speak, daemon=True)
            thread.start()
            return "Воспроизведение начато асинхронно"
        else:
            speak()
            return "Воспроизведение завершено"
    
    def _speak_with_elevenlabs(self, text: str, play_async: bool) -> Optional[str]:
        """Заглушка для ElevenLabs API"""
        if not self.config.elevenlabs_api_key:
            print("🔧 ElevenLabs API ключ не настроен, используется pyttsx3")
            return self._speak_with_pyttsx3(text, play_async)
        
        # TODO: Реализация ElevenLabs API
        print("🔧 ElevenLabs API пока не реализован, используется pyttsx3")
        return self._speak_with_pyttsx3(text, play_async)
    
    def _speak_with_local_tts(self, text: str, play_async: bool) -> Optional[str]:
        """Заглушка для локального TTS"""
        # TODO: Интеграция с локальной TTS моделью
        print("🔧 Локальный TTS пока не реализован, используется pyttsx3")
        return self._speak_with_pyttsx3(text, play_async)

class AudioHandler:
    """Основной класс для работы с аудио"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = AudioConfig(config_path)
        self.speech_recognizer = SpeechRecognizer(self.config)
        self.tts = TextToSpeech(self.config)
        
        # Для потоковой записи голоса
        self.is_listening = False
        self.audio_stream = None
        self.audio_frames = []
        self.last_speech_time = None
        
        # Пороги времени и громкости (загружаются из config.py)
        self.silence_threshold = app_config.SILENCE_THRESHOLD
        self.inactivity_timeout = app_config.INACTIVITY_TIMEOUT
        self.volume_threshold = app_config.VOICE_VOLUME_THRESHOLD
        
        self.recording_thread = None
        self.pyaudio_instance = None
        
        print("🎵 Аудио модуль инициализирован")
        self.print_status()
    
    def print_status(self):
        """Вывод статуса аудио модуля"""
        print(f"🎤 Распознавание речи: {'✅' if self.config.stt_enabled else '❌'} ({self.config.stt_engine})")
        print(f"🔊 Синтез речи: {'✅' if self.config.tts_enabled else '❌'} ({self.config.tts_engine})")
    
    def listen_for_question(self, timeout: Optional[int] = None) -> Optional[str]:
        """
        Слушает вопрос пользователя и возвращает распознанный текст
        """
        return self.speech_recognizer.listen_for_speech(timeout)
    
    def speak_answer(self, answer: str, play_async: Optional[bool] = None) -> Optional[str]:
        """
        Озвучивает ответ
        """
        actual_async = play_async if play_async is not None else self.config.auto_play
        return self.tts.speak_text(answer, actual_async)
    
    def is_speech_recognition_available(self) -> bool:
        """Проверка доступности распознавания речи"""
        return self.config.stt_enabled
    
    def is_text_to_speech_available(self) -> bool:
        """Проверка доступности синтеза речи"""
        return self.config.tts_enabled
    
    def toggle_stt(self, enabled: bool):
        """Включение/отключение распознавания речи"""
        self.config.stt_enabled = enabled
        print(f"🎤 Распознавание речи: {'включено' if enabled else 'отключено'}")
    
    def toggle_tts(self, enabled: bool):
        """Включение/отключение синтеза речи"""
        self.config.tts_enabled = enabled
        print(f"🔊 Синтез речи: {'включен' if enabled else 'отключен'}")
    
    def start_listening(self) -> Dict[str, Any]:
        """
        Начинает непрерывную запись голоса с автоматическим определением пауз
        
        Логика:
        - 2 секунды молчания → конец предложения, отправка текста
        - 5 секунд бездействия → полное отключение записи
        - После ответа нейросети автоматически возобновляется запись
        """
        if not self.config.stt_enabled or not self.speech_recognizer.recognizer:
            return {'status': 'error', 'message': 'Распознавание речи недоступно'}
        
        if self.is_listening:
            return {'status': 'already_listening'}
        
        if np is None:
            return {'status': 'error', 'message': 'numpy не установлен'}
        
        if pyaudio is None:
            return {'status': 'error', 'message': 'pyaudio не установлен'}
        
        try:
            import time
            
            self.is_listening = True
            self.audio_frames = []
            self.last_speech_time = time.time()
            
            if pyaudio is None:
                raise Exception("PyAudio не установлен")
            
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Открываем аудиопоток
            self.audio_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size
            )
            
            print("🎤 Начата запись голоса (2 сек молчания = отправка, 5 сек = отключение)")
            
            def recording_worker():
                """Фоновый поток для непрерывной записи"""
                import time
                
                silence_start = None
                has_speech = False
                
                while self.is_listening:
                    try:
                        # Чтение аудио данных
                        audio_data = self.audio_stream.read(self.config.chunk_size, exception_on_overflow=False)
                        self.audio_frames.append(audio_data)
                        
                        # Определение уровня громкости (простая эвристика)
                        audio_array = np.frombuffer(audio_data, dtype=np.int16)
                        volume = np.abs(audio_array).mean()
                        
                        current_time = time.time()
                        
                        # Если громкость выше порога - есть речь
                        if volume > self.volume_threshold:
                            has_speech = True
                            silence_start = None
                            self.last_speech_time = current_time
                        else:
                            # Тишина
                            if has_speech and silence_start is None:
                                silence_start = current_time
                            
                            if silence_start:
                                silence_duration = current_time - silence_start
                                
                                # 2 секунды тишины после речи = конец предложения
                                if silence_duration >= self.silence_threshold and has_speech:
                                    print("✅ Обнаружена пауза 2 сек - распознавание...")
                                    self._process_recorded_audio()
                                    
                                    # Сброс для следующей фразы
                                    self.audio_frames = []
                                    has_speech = False
                                    silence_start = None
                                    # Обновляем last_speech_time чтобы дать пользователю ещё 5 секунд
                                    self.last_speech_time = current_time
                        
                        # Проверка таймаута бездействия (5 секунд)
                        if current_time - self.last_speech_time > self.inactivity_timeout:
                            print("⏰ Таймаут 5 секунд - отключение записи")
                            self.stop_listening()
                            break
                        
                        time.sleep(0.01)  # Небольшая задержка
                        
                    except Exception as e:
                        print(f"❌ Ошибка записи: {e}")
                        break
                
                print("🛑 Запись завершена")
            
            # Запуск потока записи
            self.recording_thread = threading.Thread(target=recording_worker, daemon=True)
            self.recording_thread.start()
            
            return {'status': 'started', 'message': 'Запись началась'}
            
        except Exception as e:
            print(f"❌ Ошибка старта записи: {e}")
            self.is_listening = False
            return {'status': 'error', 'message': str(e)}
    
    def _process_recorded_audio(self):
        """Обработка записанного аудио (распознавание)"""
        if not self.audio_frames or len(self.audio_frames) < 10:
            # Пропускаем если фреймов мало (< 0.25 сек) - это артефакты
            return
        
        try:
            # Собираем все фреймы в один WAV
            import wave
            import io
            
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit audio = 2 bytes
                wf.setframerate(self.config.sample_rate)
                wf.writeframes(b''.join(self.audio_frames))
            
            wav_buffer.seek(0)
            
            # Распознавание через speech_recognition
            if sr and self.speech_recognizer.recognizer:
                audio_data = sr.AudioData(wav_buffer.read(), self.config.sample_rate, 2)
                
                try:
                    text = self.speech_recognizer.recognizer.recognize_google(
                        audio_data, 
                        language=self.config.stt_language
                    )
                    
                    if text:
                        print(f"🗣️  Распознано: {text}")
                        # TODO: Отправить текст в веб-интерфейс через callback
                        self._on_text_recognized(text)
                    
                except sr.UnknownValueError:
                    print("❓ Речь не распознана")
                except sr.RequestError as e:
                    print(f"❌ Ошибка API: {e}")
        
        except Exception as e:
            print(f"❌ Ошибка обработки аудио: {e}")
    
    def _on_text_recognized(self, text: str):
        """Callback для обработки распознанного текста"""
        # Этот метод будет вызываться когда текст распознан
        # В Flask endpoint можно будет добавить websocket для отправки текста клиенту
        pass
    
    def stop_listening(self) -> Optional[str]:
        """Остановка записи голоса"""
        if not self.is_listening:
            return None
        
        self.is_listening = False
        
        try:
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None
            
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
                self.pyaudio_instance = None
            
            print("🛑 Запись остановлена")
            
            # НЕ обрабатываем оставшиеся фреймы - они уже обработаны или это таймаут
            # (обработка происходит в recording_worker при паузе 2 сек)
            
            return "Запись остановлена"
            
        except Exception as e:
            print(f"❌ Ошибка остановки: {e}")
            return None
    
    def resume_listening(self):
        """Возобновление записи после ответа нейросети"""
        if not self.is_listening:
            return self.start_listening()
        return {'status': 'already_listening'}

# Пример использования
if __name__ == "__main__":
    print("🎵 Тестирование аудио модуля...")
    
    audio_handler = AudioHandler()
    
    if audio_handler.is_speech_recognition_available():
        print("\n🎤 Тест распознавания речи:")
        text = audio_handler.listen_for_question(timeout=5)
        if text:
            print(f"Распознано: {text}")
            
            if audio_handler.is_text_to_speech_available():
                print("\n🔊 Тест синтеза речи:")
                audio_handler.speak_answer(f"Вы сказали: {text}", play_async=False)
    
    print("\n✅ Тестирование завершено!")