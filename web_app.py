"""
Flask Web Application для AI-консультанта Транснефть
Современный веб-интерфейс с поддержкой streaming, голоса и камеры
"""

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import json
import time
from pathlib import Path
import sys

# Добавляем путь к main.py
sys.path.insert(0, str(Path(__file__).parent))

from main import ThinkingLLM, LLMConfig
from audio_handler import AudioHandler
from vision_handler import VisionHandler
from chat_logger import ChatLogger

app = Flask(__name__)
app.config['SECRET_KEY'] = 'transneft-ai-consultant-2025'

# Инициализация конфигурации и компонентов
config = LLMConfig()  # Используем конфиг по умолчанию
llm = ThinkingLLM(config)
audio_handler = AudioHandler()
vision_handler = VisionHandler()

# Инициализация логгера для web-сессии
chat_logger = ChatLogger(session_name="web_session")

# Глобальное хранилище для распознанного текста
recognized_text_queue = []
gesture_text_queue = []

def on_text_recognized_callback(text: str):
    """Callback для обработки распознанного текста"""
    global recognized_text_queue
    recognized_text_queue.append(text)
    print(f"📝 Добавлен текст в очередь: {text}")

def on_gesture_confirmed_callback(text: str):
    """Callback для обработки подтверждённых жестов"""
    global gesture_text_queue
    gesture_text_queue.append(text)
    print(f"👋 Добавлен жест в очередь: {text}")

# Устанавливаем callback в audio_handler
audio_handler._on_text_recognized = on_text_recognized_callback

# Устанавливаем callback для жестов
vision_handler.set_gesture_callback(on_gesture_confirmed_callback)

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Эндпоинт для чата с streaming"""
    data = request.json
    message = data.get('message', '')
    source = data.get('source', 'text')  # text/voice/gesture
    
    if not message:
        return jsonify({'error': 'Пустое сообщение'}), 400
    
    # Логируем сообщение пользователя
    chat_logger.log_user_message(message, source)
    
    # Приостанавливаем распознавание жестов на время обработки
    gesture_was_active = vision_handler.is_vision_active and not vision_handler.is_paused
    if gesture_was_active:
        print("⏸️ Приостанавливаем распознавание жестов на время обработки AI")
        vision_handler.pause_analysis()
    
    def generate():
        """Генератор для streaming ответа"""
        try:
            # Генерируем ответ
            response_dict = llm.generate_response(message)
            
            # Извлекаем thinking и answer из словаря
            thinking_content = response_dict.get('thinking', '')
            final_answer = response_dict.get('answer', '')
            
            # Логируем thinking и ответ
            chat_logger.log_assistant_thinking(thinking_content)
            chat_logger.log_assistant_answer(final_answer)
            
            # Отправляем thinking
            if thinking_content:
                yield f"data: {json.dumps({'type': 'thinking', 'content': thinking_content})}\n\n"
                time.sleep(0.1)
            
            # Отправляем ответ по предложениям
            sentences = final_answer.split('. ')
            accumulated = ""
            
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    accumulated += sentence
                    if i < len(sentences) - 1:
                        accumulated += '. '
                    
                    yield f"data: {json.dumps({'type': 'answer', 'content': accumulated})}\n\n"
                    time.sleep(0.1)
            
            # Завершение
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        finally:
            # Возобновляем распознавание жестов после обработки
            if gesture_was_active:
                print("▶️ Возобновляем распознавание жестов после обработки AI")
                vision_handler.resume_analysis()
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/api/voice/start', methods=['POST'])
def start_voice():
    """
    Запуск распознавания голоса с автоматическим определением пауз
    - 2 сек молчания → отправка текста нейросети
    - 5 сек бездействия → полное отключение
    """
    try:
        result = audio_handler.start_listening()
        return jsonify(result)
    except Exception as e:
        print(f"❌ Ошибка старта голоса: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/voice/stop', methods=['POST'])
def stop_voice():
    """Остановка распознавания голоса"""
    try:
        text = audio_handler.stop_listening()
        return jsonify({'status': 'stopped', 'text': text or ''})
    except Exception as e:
        print(f"❌ Ошибка остановки голоса: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/voice/resume', methods=['POST'])
def resume_voice():
    """Возобновление записи голоса после ответа нейросети"""
    try:
        result = audio_handler.resume_listening()
        return jsonify(result)
    except Exception as e:
        print(f"❌ Ошибка возобновления голоса: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/voice/get-text', methods=['GET'])
def get_recognized_text():
    """
    Получение распознанного текста из очереди (long polling)
    Клиент периодически опрашивает этот endpoint
    """
    global recognized_text_queue
    
    try:
        # Проверяем активна ли запись
        if not audio_handler.is_listening:
            return jsonify({'status': 'stopped', 'message': 'Запись остановлена'})
        
        if recognized_text_queue:
            text = recognized_text_queue.pop(0)
            return jsonify({'status': 'text_available', 'text': text})
        else:
            return jsonify({'status': 'no_text'})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/tts/speak', methods=['POST'])
def speak_text():
    """
    Озвучивание текста через TTS
    Принимает JSON: {"text": "текст для озвучки"}
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'status': 'error', 'error': 'Пустой текст'}), 400
        
        # Очистка текста от эмодзи и лишних символов
        import re
        # Удаляем эмодзи (Unicode диапазоны)
        text_cleaned = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+', '', text)
        # Удаляем множественные восклицательные знаки
        text_cleaned = re.sub(r'!+', '.', text_cleaned)
        # Удаляем множественные вопросительные знаки
        text_cleaned = re.sub(r'\?+', '?', text_cleaned)
        # Удаляем лишние пробелы
        text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip()
        
        if not text_cleaned:
            return jsonify({'status': 'skipped', 'message': 'Текст состоит только из символов'})
        
        print(f"🔊 Озвучивание: {text_cleaned}")
        
        # Запускаем озвучку асинхронно
        result = audio_handler.speak_answer(text_cleaned, play_async=True)
        
        return jsonify({'status': 'started', 'message': result or 'Озвучка запущена'})
        
    except Exception as e:
        print(f"❌ Ошибка озвучки: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/gesture/get-text', methods=['GET'])
def get_gesture_text():
    """
    Получение распознанного текста из жестов (long polling)
    Клиент периодически опрашивает этот endpoint
    """
    global gesture_text_queue
    
    try:
        # Проверяем активна ли камера (может быть остановлена по таймауту)
        camera_handler = vision_handler.camera_handler
        if not camera_handler.capture_thread or not camera_handler.capture_thread.is_alive():
            return jsonify({'status': 'stopped', 'message': 'Камера остановлена (таймаут 7 сек)'})
        
        if gesture_text_queue:
            text = gesture_text_queue.pop(0)
            return jsonify({'status': 'text_available', 'text': text})
        else:
            return jsonify({'status': 'no_text'})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """Запуск камеры и анализа жестов"""
    try:
        if vision_handler.start_real_time_analysis():
            return jsonify({'status': 'started', 'message': 'Камера и анализ жестов запущены'})
        else:
            return jsonify({'status': 'error', 'message': 'Не удалось запустить камеры'}), 500
    except Exception as e:
        print(f"❌ Ошибка запуска камеры: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """Остановка камеры и анализа жестов"""
    try:
        vision_handler.stop_real_time_analysis()
        return jsonify({'status': 'stopped', 'message': 'Камера остановлена'})
    except Exception as e:
        print(f"❌ Ошибка остановки камеры: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/camera/frame')
def get_camera_frame():
    """Получение кадра с камеры в формате base64"""
    try:
        frame = vision_handler.get_current_frame()
        if frame is not None:
            # Конвертируем numpy array в JPEG
            import cv2
            import base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
            
            return jsonify({
                'status': 'success',
                'frame': frame_base64,
                'format': 'base64_jpeg'
            })
        return jsonify({'status': 'error', 'error': 'Нет кадра'}), 404
    except Exception as e:
        print(f"❌ Ошибка получения кадра: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_context():
    """Сброс контекста (истории диалога)"""
    try:
        llm.clear_history()
        return jsonify({'status': 'success', 'message': 'Контекст сброшен'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Запуск Web-приложения AI-консультанта Транснефть...")
    print("📱 Откройте браузер: http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
