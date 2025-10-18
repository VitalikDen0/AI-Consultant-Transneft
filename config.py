"""
🔧 Конфигурация AI-консультанта Транснефть

Этот файл содержит ВСЕ настройки системы: от путей к моделям до параметров GPU и аудио.
Теперь config.json больше не нужен - всё здесь!
"""

# ════════════════════════════════════════════════════════════════════════════════
# 🗂️ МОДЕЛЬ LLM (Пути и технические параметры)
# ════════════════════════════════════════════════════════════════════════════════

# Путь к файлу модели (GGUF)
MODEL_PATH = "J:/models-LM Studio/mradermacher/Huihui-Qwen3-4B-Thinking-2507-abliterated-GGUF/Huihui-Qwen3-4B-Thinking-2507-abliterated.Q4_K_S.gguf"

# Размер контекста (в токенах)
# 32768 = 32K контекст, 65536 = 64K, 131072 = 128K
CONTEXT_SIZE = 32768

# Количество слоёв модели на GPU
# -1 = все слои на GPU (максимальная скорость)
# 0 = только CPU (медленно, но экономит VRAM)
# 35 = частичная выгрузка (баланс)
GPU_LAYERS = -1

# Основной GPU (если несколько видеокарт)
# 0 = первая GPU, 1 = вторая и т.д.
MAIN_GPU = 0

# Размер батча (batch size)
# Больше = быстрее, но больше VRAM
BATCH_SIZE = 512

# ════════════════════════════════════════════════════════════════════════════════
# 💾 ПАМЯТЬ И ОПТИМИЗАЦИЯ
# ════════════════════════════════════════════════════════════════════════════════

# Memory-mapped файлы (mmap) - экономит VRAM, загружает модель с диска
USE_MMAP = True

# Memory lock (mlock) - блокирует память в RAM (не рекомендуется, съедает много памяти)
USE_MLOCK = False

# Тип квантизации KV кэша для ключей (keys)
# Опции: 'f32', 'f16', 'q8_0', 'q8_1', 'q5_0', 'q5_1', 'q4_0', 'q4_1'
# q8_0 = оптимальный баланс (качество/скорость/память)
KV_CACHE_TYPE_K = "q8_0"

# Тип квантизации KV кэша для значений (values)
KV_CACHE_TYPE_V = "q8_0"

# Flash Attention (ускоряет генерацию)
FLASH_ATTENTION = True

# RoPE scaling type (для больших контекстов)
# -1 = авто, 0 = нет, 1 = linear, 2 = yarn
ROPE_SCALING_TYPE = -1

# Split mode для multi-GPU
# 0 = none, 1 = layer, 2 = row
SPLIT_MODE = 1

# NUMA оптимизация (для серверов с несколькими CPU)
NUMA = False


# ════════════════════════════════════════════════════════════════════════════════
# 🤖 ГЕНЕРАЦИЯ ТЕКСТА (LLM)
# ════════════════════════════════════════════════════════════════════════════════

# Температура генерации (0.0 = детерминированно, 1.0 = креативно)
# Рекомендуется: 0.6-0.8 для профессионального общения (0.0 на продакшн (только если не нужна креативность) и seed фиксированный)
TEMPERATURE = 0.6

# Top-p sampling (nucleus sampling)
# Чем меньше, тем более предсказуемые ответы (0.9-0.95 оптимально)
TOP_P = 0.95

# Top-k sampling
# Количество лучших токенов для выбора (20-40 оптимально)
TOP_K = 20

# Минимальная вероятность токена (min-p)
MIN_P = 0.0

# Максимальная длина ответа (в токенах)
# 8192 = ~6000 слов на русском (-1 = неограниченно)
MAX_TOKENS = -1

# Максимальная длина "thinking" блока (внутренние размышления) (Не работает для моей модели Qwen3-4B-Thinking, актуально для GPTOss)
# 1024 = ~750 слов
MAX_THINKING_TOKENS = 1024


# ════════════════════════════════════════════════════════════════════════════════
# 🎤 ГОЛОСОВОЙ ВВОД (Speech-to-Text)
# ════════════════════════════════════════════════════════════════════════════════

# Включить/выключить распознавание речи
STT_ENABLED = True

# Движок распознавания: 'google', 'whisper_api', 'local_whisper'
STT_ENGINE = "google"

# Язык распознавания (ISO код)
STT_LANGUAGE = "ru-RU"

# API ключ для Whisper API (OpenAI) - оставить пустым если не используется
WHISPER_API_KEY = ""

# URL для Whisper API
WHISPER_API_URL = "https://api.openai.com/v1/audio/transcriptions"

# Порог громкости для определения речи (0-32768)
# Чем выше, тем меньше чувствительность к тихим звукам
VOICE_VOLUME_THRESHOLD = 500

# Порог тишины для окончания предложения (секунды)
# Пользователь замолчал → текст распознаётся и отправляется
SILENCE_THRESHOLD = 2.0

# Таймаут бездействия (секунды)
# Если 5 секунд нет речи → запись автоматически отключается
INACTIVITY_TIMEOUT = 5.0

# Частота дискретизации аудио (Hz)
# 16000 Hz оптимально для речи (не менять без необходимости)
AUDIO_SAMPLE_RATE = 16000

# Размер буфера аудио (сэмплы)
# Меньше = быстрее отклик, больше = стабильнее
AUDIO_CHUNK_SIZE = 1024


# ════════════════════════════════════════════════════════════════════════════════
# 🔊 ОЗВУЧИВАНИЕ ОТВЕТОВ (Text-to-Speech)
# ════════════════════════════════════════════════════════════════════════════════

# Включить/выключить синтез речи
TTS_ENABLED = True

# Движок синтеза: 'pyttsx3', 'elevenlabs', 'azure', 'google'
TTS_ENGINE = "pyttsx3"

# Голос (для pyttsx3: 'russian', для других движков - ID голоса)
TTS_VOICE = "russian"

# Скорость речи (слов в минуту)
# 120-150 = комфортно, 150-180 = быстро, 180+ = очень быстро
# ВНИМАНИЕ: Реальная скорость будет увеличена на 20% (TTS_RATE * 1.2)
# Например: 150 → 180 слов/мин
TTS_RATE = 150

# Предпочтение мужского голоса (True = искать мужской, False = любой)
TTS_PREFER_MALE = True

# Громкость (0.0 - 1.0)
TTS_VOLUME = 0.9

# Автоматически озвучивать ответы (если кнопка 🔊 активна)
TTS_AUTO_PLAY = True

# API ключ для ElevenLabs (если используется)
ELEVENLABS_API_KEY = ""

# ID голоса ElevenLabs
ELEVENLABS_VOICE_ID = ""


# ════════════════════════════════════════════════════════════════════════════════
# 📹 КАМЕРА И ЖЕСТЫ (Vision - опционально)
# ════════════════════════════════════════════════════════════════════════════════

# Включить/выключить распознавание жестов
VISION_ENABLED = False

# ID камеры (0 = первая камера, 1 = вторая и т.д.)
CAMERA_DEVICE_ID = 0

# FPS камеры (кадров в секунду)
# 16 FPS оптимально для жестов (не перегружает CPU)
CAMERA_FPS = 16

# Разрешение камеры [ширина, высота]
CAMERA_RESOLUTION = [640, 480]

# Уверенность детекции жестов (0.0-1.0)
# Чем выше, тем меньше ложных срабатываний
GESTURE_CONFIDENCE = 0.6

# Интервал анализа жестов (секунды)
# 0.1 = проверка каждые 100ms
GESTURE_ANALYSIS_INTERVAL = 0.1

# Время подтверждения жеста (секунды)
# Жест должен повторяться в течение этого времени
GESTURE_CONFIRMATION_TIME = 1.5

# Таймаут бездействия для жестов (секунды)
# Если нет жестов в течение этого времени - автоматическая остановка
GESTURE_INACTIVITY_TIMEOUT = 7.0

# MediaPipe настройки
MEDIAPIPE_MODEL_COMPLEXITY = 1  # 0 = lite, 1 = full, 2 = heavy
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.7
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5
MEDIAPIPE_MAX_NUM_HANDS = 2

# Обработка кадров
FRAME_SKIP = 2  # Пропускать каждый N-й кадр для экономии CPU

# Детекция жестов
GESTURE_DETECTION_ENABLED = True

# Функции Vision
REAL_TIME_ANALYSIS = True
GESTURE_RECOGNITION = True
HAND_TRACKING = True
POSE_DETECTION = False


# ════════════════════════════════════════════════════════════════════════════════
# 🗄️ ИСТОРИЯ ДИАЛОГА
# ════════════════════════════════════════════════════════════════════════════════

# Максимальное количество сообщений в истории (для контекста)
# Чем больше, тем лучше память, но больше токенов расходуется
MAX_HISTORY_MESSAGES = 10

# Хранить всю историю в памяти (True) или только последние N сообщений (False)
KEEP_FULL_HISTORY = False


# ════════════════════════════════════════════════════════════════════════════════
# 🐛 ОТЛАДКА
# ════════════════════════════════════════════════════════════════════════════════

# Показывать подробные логи (True) или только важные (False)
VERBOSE_LOGGING = True

# Показывать "thinking" блок пользователю (для отладки)
SHOW_THINKING_TO_USER = True


# ════════════════════════════════════════════════════════════════════════════════
# ⚙️ СЛУЖЕБНЫЕ ФУНКЦИИ
# ════════════════════════════════════════════════════════════════════════════════

def get_model_config() -> dict:
    """Получить настройки модели и памяти"""
    return {
        'path': MODEL_PATH,
        'context_size': CONTEXT_SIZE,
        'gpu_layers': GPU_LAYERS,
        'main_gpu': MAIN_GPU,
        'batch_size': BATCH_SIZE,
        'use_mmap': USE_MMAP,
        'use_mlock': USE_MLOCK,
        'kv_cache_type_k': KV_CACHE_TYPE_K,
        'kv_cache_type_v': KV_CACHE_TYPE_V,
        'flash_attention': FLASH_ATTENTION,
        'rope_scaling_type': ROPE_SCALING_TYPE,
        'split_mode': SPLIT_MODE,
        'numa': NUMA
    }


def get_generation_config() -> dict:
    """Получить настройки генерации текста для LLM"""
    return {
        'temperature': TEMPERATURE,
        'top_p': TOP_P,
        'top_k': TOP_K,
        'min_p': MIN_P,
        'max_tokens': MAX_TOKENS,
        'max_thinking_tokens': MAX_THINKING_TOKENS
    }


def get_audio_config() -> dict:
    """Получить настройки аудио (STT/TTS)"""
    return {
        'stt_enabled': STT_ENABLED,
        'stt_engine': STT_ENGINE,
        'stt_language': STT_LANGUAGE,
        'whisper_api_key': WHISPER_API_KEY,
        'whisper_api_url': WHISPER_API_URL,
        'volume_threshold': VOICE_VOLUME_THRESHOLD,
        'silence_threshold': SILENCE_THRESHOLD,
        'inactivity_timeout': INACTIVITY_TIMEOUT,
        'sample_rate': AUDIO_SAMPLE_RATE,
        'chunk_size': AUDIO_CHUNK_SIZE,
        'tts_enabled': TTS_ENABLED,
        'tts_engine': TTS_ENGINE,
        'tts_voice': TTS_VOICE,
        'tts_rate': TTS_RATE,
        'tts_volume': TTS_VOLUME,
        'tts_auto_play': TTS_AUTO_PLAY,
        'elevenlabs_api_key': ELEVENLABS_API_KEY,
        'elevenlabs_voice_id': ELEVENLABS_VOICE_ID
    }


def get_vision_config() -> dict:
    """Получить настройки камеры и жестов"""
    return {
        'enabled': VISION_ENABLED,
        'camera_device_id': CAMERA_DEVICE_ID,
        'camera_fps': CAMERA_FPS,
        'camera_resolution': CAMERA_RESOLUTION,
        'gesture_confidence': GESTURE_CONFIDENCE,
        'gesture_analysis_interval': GESTURE_ANALYSIS_INTERVAL,
        'gesture_confirmation_time': GESTURE_CONFIRMATION_TIME,
        'gesture_inactivity_timeout': GESTURE_INACTIVITY_TIMEOUT,
        'mediapipe_model_complexity': MEDIAPIPE_MODEL_COMPLEXITY,
        'mediapipe_min_detection_confidence': MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
        'mediapipe_min_tracking_confidence': MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        'mediapipe_max_num_hands': MEDIAPIPE_MAX_NUM_HANDS,
        'frame_skip': FRAME_SKIP,
        'gesture_detection_enabled': GESTURE_DETECTION_ENABLED,
        'real_time_analysis': REAL_TIME_ANALYSIS,
        'gesture_recognition': GESTURE_RECOGNITION,
        'hand_tracking': HAND_TRACKING,
        'pose_detection': POSE_DETECTION
    }


def get_history_config() -> dict:
    """Получить настройки истории диалога"""
    return {
        'max_messages': MAX_HISTORY_MESSAGES,
        'keep_full_history': KEEP_FULL_HISTORY
    }


def print_current_config():
    """Вывести текущие настройки в консоль"""
    print("\n" + "="*80)
    print("⚙️  ТЕКУЩАЯ КОНФИГУРАЦИЯ")
    print("="*80)
    
    print("\n🗂️  МОДЕЛЬ:")
    print(f"  Путь: {MODEL_PATH}")
    print(f"  Контекст: {CONTEXT_SIZE} токенов")
    print(f"  GPU слои: {GPU_LAYERS}")
    print(f"  Batch size: {BATCH_SIZE}")
    
    print("\n💾 ПАМЯТЬ:")
    print(f"  MMAP: {USE_MMAP}")
    print(f"  MLOCK: {USE_MLOCK}")
    print(f"  KV cache: {KV_CACHE_TYPE_K}/{KV_CACHE_TYPE_V}")
    print(f"  Flash Attention: {FLASH_ATTENTION}")
    
    print("\n🤖 ГЕНЕРАЦИЯ ТЕКСТА:")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Top-p: {TOP_P}")
    print(f"  Top-k: {TOP_K}")
    print(f"  Max tokens: {MAX_TOKENS}")
    
    print("\n🎤 ГОЛОСОВОЙ ВВОД:")
    print(f"  Включён: {STT_ENABLED}")
    print(f"  Движок: {STT_ENGINE}")
    print(f"  Язык: {STT_LANGUAGE}")
    print(f"  Порог молчания: {SILENCE_THRESHOLD} сек")
    print(f"  Таймаут: {INACTIVITY_TIMEOUT} сек")
    print(f"  Громкость: {VOICE_VOLUME_THRESHOLD}")
    
    print("\n🔊 ОЗВУЧИВАНИЕ:")
    print(f"  Включено: {TTS_ENABLED}")
    print(f"  Движок: {TTS_ENGINE}")
    print(f"  Скорость: {TTS_RATE} слов/мин")
    print(f"  Громкость: {TTS_VOLUME}")
    
    print("\n📹 КАМЕРА:")
    print(f"  Включена: {VISION_ENABLED}")
    print(f"  Device ID: {CAMERA_DEVICE_ID}")
    print(f"  FPS: {CAMERA_FPS}")
    print(f"  Разрешение: {CAMERA_RESOLUTION}")
    
    print("="*80 + "\n")


# ════════════════════════════════════════════════════════════════════════════════
# 📝 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Показать текущие настройки
    print_current_config()
    
    # Примеры получения конфигов
    gen_config = get_generation_config()
    print(f"Конфиг генерации: {gen_config}")
    
    audio_config = get_audio_config()
    print(f"Конфиг аудио: {audio_config}")
