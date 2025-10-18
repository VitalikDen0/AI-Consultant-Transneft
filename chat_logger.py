"""
Chat Logger - Логирование диалогов в файл
Используется как в web режиме, так и в консольном
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class ChatLogger:
    """Класс для логирования чат-диалогов"""
    
    def __init__(self, log_dir: str = "chat_logs", session_name: Optional[str] = None):
        """
        Инициализация логгера
        
        Args:
            log_dir: Директория для логов
            session_name: Имя сессии (если None - генерируется автоматически)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Генерация имени файла
        if session_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"chat_{timestamp}"
        
        self.log_file = self.log_dir / f"{session_name}.log"
        self.enabled = True
        
        # Создание заголовка лога
        self._write_header()
    
    def _write_header(self):
        """Запись заголовка лога"""
        header = f"""{'=' * 80}
CHAT LOG - AI-Консультант ПАО «Транснефть»
Дата начала сессии: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'=' * 80}

"""
        self._write_to_file(header)
    
    def _write_to_file(self, text: str):
        """Запись текста в файл"""
        if not self.enabled:
            return
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(text)
        except Exception as e:
            print(f"⚠️ Ошибка записи в лог: {e}")
    
    def log_user_message(self, message: str, source: str = "text"):
        """
        Логирование сообщения пользователя
        
        Args:
            message: Текст сообщения
            source: Источник (text/voice/gesture)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        source_emoji = {
            "text": "💬",
            "voice": "🎤",
            "gesture": "👋"
        }.get(source, "📝")
        
        log_entry = f"""
{'─' * 80}
[{timestamp}] {source_emoji} ПОЛЬЗОВАТЕЛЬ ({source}):
{message}
"""
        self._write_to_file(log_entry)
    
    def log_assistant_thinking(self, thinking: str):
        """
        Логирование размышлений AI (thinking)
        
        Args:
            thinking: Текст размышлений
        """
        if not thinking:
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        log_entry = f"""
[{timestamp}] 💭 AI РАЗМЫШЛЕНИЯ (Thinking):
{thinking}
"""
        self._write_to_file(log_entry)
    
    def log_assistant_answer(self, answer: str):
        """
        Логирование ответа AI
        
        Args:
            answer: Текст ответа
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        log_entry = f"""
[{timestamp}] 🤖 AI ОТВЕТ:
{answer}
"""
        self._write_to_file(log_entry)
    
    def log_error(self, error: str):
        """
        Логирование ошибки
        
        Args:
            error: Описание ошибки
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        log_entry = f"""
[{timestamp}] ❌ ОШИБКА:
{error}
"""
        self._write_to_file(log_entry)
    
    def log_system_event(self, event: str):
        """
        Логирование системного события
        
        Args:
            event: Описание события
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        log_entry = f"""
[{timestamp}] ⚙️ СИСТЕМА:
{event}
"""
        self._write_to_file(log_entry)
    
    def close(self):
        """Закрытие лога"""
        footer = f"""
{'=' * 80}
Сессия завершена: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Лог сохранён: {self.log_file}
{'=' * 80}
"""
        self._write_to_file(footer)
        print(f"📄 Лог диалога сохранён: {self.log_file}")
