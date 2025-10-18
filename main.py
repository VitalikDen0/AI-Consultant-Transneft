#!/usr/bin/env python3
"""
LLM Chat Interface with GGUF Model Support

Настроенная система для работы с локальной GGUF моделью через llama-cpp-python.
Поддерживает thinking mode и полную выгрузку на GPU.
"""

import os
import sys
import json
import argparse
import time
from typing import Optional, Dict, Any, List
from pathlib import Path

# Установка CUDA переменной окружения для стабильности
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

try:
    from llama_cpp import Llama
    from colorama import init, Fore, Style, Back
    import psutil
    from audio_handler import AudioHandler
    from vision_handler import VisionHandler
except ImportError as e:
    print(f"Ошибка импорта зависимостей: {e}")
    print("Убедитесь, что все зависимости установлены: pip install -r requirements.txt")
    sys.exit(1)

# Инициализация colorama для Windows
init(autoreset=True)

class LLMConfig:
    """Конфигурация для LLM модели с оптимизациями"""
    
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        # Импорт настроек из config.py
        try:
            import config as app_config
        except ImportError:
            raise ImportError("❌ config.py не найден! Создайте файл config.py с настройками.")
        
        # Путь к модели (приоритет: аргумент -> config.py)
        self.model_path = model_path if model_path else app_config.MODEL_PATH
        
        # Основные параметры модели
        self.n_ctx = app_config.CONTEXT_SIZE
        self.n_gpu_layers = app_config.GPU_LAYERS
        self.main_gpu = app_config.MAIN_GPU
        self.n_batch = app_config.BATCH_SIZE
        
        # Параметры памяти
        self.use_mmap = app_config.USE_MMAP
        self.use_mlock = app_config.USE_MLOCK
        
        # KV cache настройки
        self.type_k = self._parse_kv_type(app_config.KV_CACHE_TYPE_K)
        self.type_v = self._parse_kv_type(app_config.KV_CACHE_TYPE_V)
        
        # CPU настройки
        self.n_threads = psutil.cpu_count()
        self.n_threads_batch = psutil.cpu_count()
        
        # Параметры генерации
        self.temperature = app_config.TEMPERATURE
        self.top_p = app_config.TOP_P
        self.top_k = app_config.TOP_K
        self.min_p = app_config.MIN_P
        self.max_tokens = app_config.MAX_TOKENS
        self.max_thinking_tokens = app_config.MAX_THINKING_TOKENS
        
        # Оптимизации
        self.flash_attn = app_config.FLASH_ATTENTION
        self.rope_scaling_type = app_config.ROPE_SCALING_TYPE
        self.split_mode = app_config.SPLIT_MODE
        self.numa = app_config.NUMA
        
        # Отладка
        self.verbose = app_config.VERBOSE_LOGGING
        
        print(f"{Fore.GREEN}✓ Конфигурация загружена из config.py")
    
    def _parse_kv_type(self, type_str: str) -> int:
        """Преобразование строкового типа KV кэша в цифру"""
        type_mapping = {
            'q4_0': 2,
            'q4_1': 3,
            'q5_0': 6,
            'q5_1': 7,
            'q8_0': 8,
            'q8_1': 9,
            'f16': 1,
            'f32': 0
        }
        return type_mapping.get(type_str.lower(), 8)  # По умолчанию Q8_0
    
    def print_config(self):
        """Вывод текущей конфигурации"""
        print(f"{Fore.MAGENTA}=== Конфигурация модели ===")
        print(f"{Fore.BLUE}Модель: {self.model_path}")
        print(f"{Fore.BLUE}Контекст: {self.n_ctx}")
        print(f"{Fore.BLUE}GPU слои: {self.n_gpu_layers}")
        print(f"{Fore.BLUE}Основной GPU: {self.main_gpu}")
        print(f"{Fore.BLUE}Temperature: {self.temperature}")
        print(f"{Fore.BLUE}Top P: {self.top_p}")
        print(f"{Fore.BLUE}Top K: {self.top_k}")
        print(f"{Fore.BLUE}KV cache: type_k={self.type_k}, type_v={self.type_v}")
        print(f"{Fore.BLUE}Flash attention: {self.flash_attn}")
        print(f"{Fore.BLUE}MMAP: {self.use_mmap}, MLOCK: {self.use_mlock}")
        print(f"{Fore.MAGENTA}=========================")

class ThinkingLLM:
    """Обертка для работы с thinking моделью с поддержкой истории диалога"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.llama = None
        self.conversation_history = []  # История диалога
        self.max_history_tokens = config.n_ctx // 2  # Максимум токенов для истории (половина контекста)
        
        # Загружаем базу знаний из PROMT.md
        self.knowledge_base = self._load_knowledge_base()
        
        self._initialize_model()
    
    def _load_knowledge_base(self) -> str:
        """Загрузка базы знаний из PROMT.md"""
        try:
            promt_path = Path(__file__).parent / "PROMT.md"
            if promt_path.exists():
                with open(promt_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"{Fore.GREEN}✓ База знаний загружена из PROMT.md ({len(content)} символов)")
                return content
            else:
                print(f"{Fore.YELLOW}⚠️  Файл PROMT.md не найден, используется стандартный промпт")
                return ""
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️  Ошибка загрузки базы знаний: {e}")
            return ""
    
    def _initialize_model(self):
        """Инициализация модели с настройками"""
        print(f"{Fore.YELLOW}Инициализация модели...")
        print(f"{Fore.CYAN}Модель: {self.config.model_path}")
        print(f"{Fore.CYAN}Контекст: {self.config.n_ctx} токенов")
        print(f"{Fore.CYAN}GPU слои: {self.config.n_gpu_layers} (полная выгрузка)")
        print(f"{Fore.CYAN}Основной GPU: {self.config.main_gpu}")
        print(f"{Fore.CYAN}CPU потоки: {self.config.n_threads}")
        print(f"{Fore.CYAN}KV кэш: type_k={self.config.type_k}, type_v={self.config.type_v}")
        print(f"{Fore.CYAN}KV кэш: Автоматические оптимизации")
        
        if not Path(self.config.model_path).exists():
            raise FileNotFoundError(f"Модель не найдена: {self.config.model_path}")
        
        try:
            self.llama = Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.n_ctx,
                n_batch=self.config.n_batch,  # Добавляем batch size
                n_gpu_layers=self.config.n_gpu_layers,
                main_gpu=self.config.main_gpu,
                use_mmap=self.config.use_mmap,  # True для экономии VRAM
                use_mlock=self.config.use_mlock,  # False для экономии VRAM
                n_threads=self.config.n_threads,
                n_threads_batch=self.config.n_threads_batch,
                rope_scaling_type=self.config.rope_scaling_type,
                verbose=self.config.verbose,
                split_mode=self.config.split_mode,
                numa=self.config.numa,
                # VRAM оптимизации
                offload_kqv=True,  # Выгрузка KV кэша на GPU
                logits_all=False,  # Не сохраняем все logits
                embedding=False,  # Отключаем embedding если не нужен
                # KV cache квантизация для экономии VRAM
                type_k=self.config.type_k,
                type_v=self.config.type_v,
                flash_attn=self.config.flash_attn
            )
            print(f"{Fore.GREEN}✓ Модель успешно загружена!")
            self._print_model_info()
        except Exception as e:
            error_msg = str(e)
            print(f"{Fore.RED}Ошибка загрузки модели: {error_msg}")
            
            # Специальная обработка CUDA ошибок
            if "CUDA" in error_msg or "cuda" in error_msg:
                print(f"{Fore.YELLOW}🔧 Обнаружена CUDA ошибка. Попробуем загрузить только на CPU...")
                try:
                    # Fallback на CPU
                    self.config.n_gpu_layers = 0
                    self.llama = Llama(
                        model_path=self.config.model_path,
                        n_ctx=self.config.n_ctx,
                        n_gpu_layers=0,  # Только CPU
                        use_mmap=self.config.use_mmap,
                        use_mlock=self.config.use_mlock,
                        n_threads=self.config.n_threads,
                        verbose=self.config.verbose,
                        # KV cache квантизация даже на CPU
                        type_k=self.config.type_k,
                        type_v=self.config.type_v,
                    )
                    print(f"{Fore.GREEN}✓ Модель загружена на CPU!")
                    self._print_model_info()
                    return
                except Exception as cpu_error:
                    print(f"{Fore.RED}Ошибка загрузки на CPU: {cpu_error}")
            
            raise
    
    def _print_model_info(self):
        """Вывод информации о модели"""
        if self.llama is None:
            print(f"{Fore.YELLOW}Модель не загружена")
            return
            
        try:
            # Получение метаданных модели
            metadata = getattr(self.llama, 'metadata', {})
            vocab_size = self.llama.n_vocab() if hasattr(self.llama, 'n_vocab') else 'Неизвестно'
            ctx_size = self.llama.n_ctx() if hasattr(self.llama, 'n_ctx') else 'Неизвестно'
            
            print(f"{Fore.MAGENTA}=== Информация о модели ===")
            print(f"{Fore.BLUE}Размер словаря: {vocab_size}")
            print(f"{Fore.BLUE}Контекст: {ctx_size}")
            
            # Вывод метаданных если доступны
            if metadata and isinstance(metadata, dict):
                for key, value in metadata.items():
                    if key.startswith('general.'):
                        print(f"{Fore.BLUE}{key}: {value}")
            
        except Exception as e:
            print(f"{Fore.YELLOW}Не удалось получить метаданные модели: {e}")
    
    def _trim_conversation_history(self):
        """
        Обрезает историю диалога, чтобы не превышать лимит контекста
        """
        if not self.conversation_history or self.llama is None:
            return
        
        # Системное сообщение всегда сохраняем
        system_messages = [msg for msg in self.conversation_history if msg['role'] == 'system']
        other_messages = [msg for msg in self.conversation_history if msg['role'] != 'system']
        
        # Подсчитываем токены для всех сообщений
        total_tokens = 0
        kept_messages = []
        
        # Обрабатываем сообщения в обратном порядке (сначала самые новые)
        for message in reversed(other_messages):
            try:
                message_text = f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
                message_tokens = len(self.llama.tokenize(message_text.encode('utf-8')))
                
                if total_tokens + message_tokens <= self.max_history_tokens:
                    kept_messages.insert(0, message)
                    total_tokens += message_tokens
                else:
                    break
            except Exception:
                # В случае ошибки токенизации, приблизительная оценка
                estimated_tokens = len(message['content'].split()) * 1.3
                if total_tokens + estimated_tokens <= self.max_history_tokens:
                    kept_messages.insert(0, message)
                    total_tokens += estimated_tokens
                else:
                    break
        
        # Обновляем историю
        self.conversation_history = system_messages + kept_messages
        
        if len(other_messages) > len(kept_messages):
            removed_count = len(other_messages) - len(kept_messages)
            print(f"{Fore.YELLOW}📝 Удалено {removed_count} старых сообщений из истории для экономии токенов")
    
    def add_to_history(self, role: str, content: str):
        """
        Добавляет сообщение в историю диалога
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        self._trim_conversation_history()
    
    def clear_history(self):
        """
        Очищает историю диалога (кроме системного сообщения)
        """
        system_messages = [msg for msg in self.conversation_history if msg['role'] == 'system']
        self.conversation_history = system_messages
        print(f"{Fore.GREEN}🗑️ История диалога очищена")
    
    def get_history_summary(self) -> str:
        """
        Возвращает краткую сводку истории диалога
        """
        if not self.conversation_history:
            return "История пуста"
        
        user_messages = len([msg for msg in self.conversation_history if msg['role'] == 'user'])
        assistant_messages = len([msg for msg in self.conversation_history if msg['role'] == 'assistant'])
        
        return f"История: {user_messages} вопросов, {assistant_messages} ответов"

    def _create_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """
        Создание chat template согласно предоставленному Jinja шаблону
        """
        # Упрощенная реализация Jinja шаблона
        # В полной реализации следовало бы использовать jinja2
        
        formatted_messages = []
        
        # Обработка системного сообщения
        if messages and messages[0].get('role') == 'system':
            formatted_messages.append(f"<|im_start|>system\n{messages[0]['content']}<|im_end|>")
            messages = messages[1:]
        
        # Обработка остальных сообщений
        for message in messages:
            role = message['role']
            content = message['content']
            
            if role in ['user', 'assistant']:
                formatted_messages.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        
        # Добавление начала ответа ассистента
        formatted_messages.append("<|im_start|>assistant\n")
        
        return "\n".join(formatted_messages)
    
    def generate_response(self, user_input: str) -> Dict[str, Any]:
        """
        Генерация ответа с thinking режимом и статистикой токенов
        """
        if self.llama is None:
            return {
                "thinking": "Модель не загружена",
                "answer": "Ошибка: модель не инициализирована",
                "full_response": "Model not loaded",
                "token_stats": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "tokens_per_second": 0.0}
            }
        
        # Добавляем пользовательский ввод в историю
        self.add_to_history("user", user_input)
        
        # Создаем список сообщений из истории или инициализируем системное сообщение
        if not any(msg['role'] == 'system' for msg in self.conversation_history):
            # Формируем системный промпт с базой знаний
            system_prompt = "Ты - профессиональный AI-консультант компании Транснефть. "
            system_prompt += "Твоя задача - помогать пользователям с вопросами о компании, предоставлять информацию и консультировать. "
            system_prompt += "Отвечай профессионально, дружелюбно и информативно на русском языке. "
            system_prompt += "Используй <think></think> теги для демонстрации процесса рассуждения перед финальным ответом.\n\n"
            
            # Добавляем базу знаний если она загружена
            if self.knowledge_base:
                system_prompt += "# БАЗА ЗНАНИЙ О КОМПАНИИ ТРАНСНЕФТЬ\n\n"
                system_prompt += self.knowledge_base
                system_prompt += "\n\n# КОНЕЦ БАЗЫ ЗНАНИЙ\n\n"
                system_prompt += "ВАЖНО: Используй информацию из базы знаний для ответов на вопросы о компании. "
                system_prompt += "Если информации в базе недостаточно, скажи об этом честно."
            
            self.add_to_history("system", system_prompt)
        
        # Используем всю историю диалога для генерации
        messages = self.conversation_history.copy()
        
        prompt = self._create_chat_template(messages)
        
        # Подсчет входных токенов
        try:
            input_tokens = len(self.llama.tokenize(prompt.encode('utf-8')))
        except Exception:
            input_tokens = 0
        
        print(f"{Fore.YELLOW}Генерация ответа...")
        print(f"{Fore.CYAN}📊 Входных токенов: {input_tokens}")
        print(f"{Fore.CYAN}📚 {self.get_history_summary()}")
        
        # Засекаем время начала генерации
        start_time = time.time()
        
        try:
            # Параметры генерации (настроены согласно требованиям)
            response = self.llama(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                min_p=self.config.min_p,
                stop=["<|im_end|>", "</s>"],  # Стоп токены
                echo=False,  # Не повторять промпт
                stream=False,  # Без стриминга
            )
            
            # Засекаем время окончания
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Правильная обработка ответа
            if isinstance(response, dict) and 'choices' in response:
                full_response = response['choices'][0]['text'].strip()
                
                # Получение статистики использования токенов из ответа
                usage = response.get('usage', {})
                prompt_tokens = usage.get('prompt_tokens', input_tokens)
                completion_tokens = usage.get('completion_tokens', 0)
                total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
                
                # Если completion_tokens не доступно, пытаемся подсчитать сами
                if completion_tokens == 0 and full_response:
                    try:
                        completion_tokens = len(self.llama.tokenize(full_response.encode('utf-8')))
                        total_tokens = prompt_tokens + completion_tokens
                    except Exception:
                        completion_tokens = len(full_response.split())  # Приблизительная оценка
                        total_tokens = prompt_tokens + completion_tokens
                
            else:
                # Fallback для других форматов ответа
                full_response = str(response).strip()
                prompt_tokens = input_tokens
                try:
                    completion_tokens = len(self.llama.tokenize(full_response.encode('utf-8')))
                except Exception:
                    completion_tokens = len(full_response.split())  # Приблизительная оценка
                total_tokens = prompt_tokens + completion_tokens
            
            # Вычисление скорости генерации
            tokens_per_second = completion_tokens / generation_time if generation_time > 0 else 0.0
            
            # Извлечение thinking и финального ответа
            thinking_content = ""
            final_answer = full_response
            
            if "<think>" in full_response and "</think>" in full_response:
                parts = full_response.split("<think>", 1)
                if len(parts) > 1:
                    think_part = parts[1].split("</think>", 1)
                    if len(think_part) > 1:
                        thinking_content = think_part[0].strip()
                        final_answer = think_part[1].strip()
            
            # Если thinking не найден, вся генерация считается финальным ответом
            if not thinking_content:
                thinking_content = "Reasoning process not found in response"
            
            # Добавляем ответ ассистента в историю (только финальный ответ, без thinking)
            self.add_to_history("assistant", final_answer)
            
            # Вывод в консоль
            print(f"\n{Fore.CYAN}💭 Thinking:{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}{thinking_content}{Style.RESET_ALL}")
            print(f"\n{Fore.GREEN}✅ Answer:{Style.RESET_ALL}")
            print(f"{Fore.WHITE}{final_answer}{Style.RESET_ALL}\n")
            
            # Статистика токенов
            token_stats = {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "tokens_per_second": tokens_per_second,
                "generation_time": generation_time
            }
            
            return {
                "thinking": thinking_content,
                "answer": final_answer,
                "full_response": full_response,
                "token_stats": token_stats
            }
            
        except Exception as e:
            end_time = time.time()
            generation_time = end_time - start_time
            
            return {
                "thinking": f"Ошибка при генерации: {e}",
                "answer": "Извините, произошла ошибка при генерации ответа.",
                "full_response": f"Error: {e}",
                "token_stats": {
                    "input_tokens": input_tokens,
                    "output_tokens": 0,
                    "total_tokens": input_tokens,
                    "tokens_per_second": 0.0,
                    "generation_time": generation_time
                }
            }

def print_token_stats(token_stats: Dict[str, Any]):
    """Красивый вывод статистики токенов"""
    print(f"\n{Back.MAGENTA}{Fore.WHITE} TOKEN STATISTICS {Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}┌{'─'*60}┐")
    print(f"{Fore.MAGENTA}│ {Fore.CYAN}📥 Входных токенов:  {token_stats['input_tokens']:>8} {Fore.MAGENTA}                │")
    print(f"{Fore.MAGENTA}│ {Fore.CYAN}📤 Выходных токенов: {token_stats['output_tokens']:>8} {Fore.MAGENTA}                │")
    print(f"{Fore.MAGENTA}│ {Fore.CYAN}📊 Всего токенов:    {token_stats['total_tokens']:>8} {Fore.MAGENTA}                │")
    print(f"{Fore.MAGENTA}│ {Fore.CYAN}⚡ Скорость:         {token_stats['tokens_per_second']:>8.2f} т/сек {Fore.MAGENTA}      │")
    print(f"{Fore.MAGENTA}│ {Fore.CYAN}⏱️  Время генерации: {token_stats['generation_time']:>8.2f} сек {Fore.MAGENTA}       │")
    print(f"{Fore.MAGENTA}└{'─'*60}┘")

def print_separator():
    """Печать разделителя"""
    print(f"{Fore.CYAN}{'='*80}")

def print_thinking(thinking_text: str):
    """Красивый вывод thinking секции"""
    print(f"\n{Back.BLUE}{Fore.WHITE} THINKING PROCESS {Style.RESET_ALL}")
    print(f"{Fore.BLUE}┌{'─'*78}┐")
    
    # Разделение на строки и форматирование
    lines = thinking_text.split('\n')
    for line in lines:
        # Ограничение длины строки
        if len(line) > 76:
            words = line.split(' ')
            current_line = ""
            for word in words:
                if len(current_line + word) > 76:
                    print(f"{Fore.BLUE}│ {Fore.CYAN}{current_line:<76}{Fore.BLUE} │")
                    current_line = word + " "
                else:
                    current_line += word + " "
            if current_line.strip():
                print(f"{Fore.BLUE}│ {Fore.CYAN}{current_line.strip():<76}{Fore.BLUE} │")
        else:
            print(f"{Fore.BLUE}│ {Fore.CYAN}{line:<76}{Fore.BLUE} │")
    
    print(f"{Fore.BLUE}└{'─'*78}┘")

def print_answer(answer_text: str):
    """Красивый вывод финального ответа"""
    print(f"\n{Back.GREEN}{Fore.WHITE} FINAL ANSWER {Style.RESET_ALL}")
    print(f"{Fore.GREEN}{answer_text}")

def main():
    """Основная функция чата"""
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='LLM Chat Interface with Thinking Mode and Audio Support')
    parser.add_argument('--model', '-m', type=str, help='Путь к модели GGUF')
    parser.add_argument('--test', '-t', action='store_true', help='Режим тестирования (без загрузки модели)')
    parser.add_argument('--no-audio', action='store_true', help='Отключить аудио функции')
    parser.add_argument('--no-stt', action='store_true', help='Отключить распознавание речи')
    parser.add_argument('--no-tts', action='store_true', help='Отключить синтез речи')
    args = parser.parse_args()
    
    try:
        print(f"{Fore.MAGENTA}🤖 LLM Chat Interface with Thinking Mode & Audio")
        print(f"{Fore.MAGENTA}Модель: Huihui-Qwen3-4B-Thinking")
        print_separator()
        
        if args.test:
            print(f"{Fore.YELLOW}🧪 Режим тестирования - модель не загружается")
            print(f"{Fore.GREEN}✓ Все зависимости установлены корректно!")
            print(f"{Fore.BLUE}Для реального запуска используйте без флага --test")
            return
        
        # Инициализация аудио модуля
        audio_handler = None
        if not args.no_audio:
            try:
                audio_handler = AudioHandler()
                if args.no_stt:
                    audio_handler.toggle_stt(False)
                if args.no_tts:
                    audio_handler.toggle_tts(False)
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️  Аудио модуль недоступен: {e}")
                print(f"{Fore.CYAN}💡 Для использования аудио установите: pip install pyaudio speechrecognition pyttsx3")
                audio_handler = None
        
        # Инициализация vision модуля
        vision_handler = None
        try:
            vision_handler = VisionHandler()
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️  Vision модуль недоступен: {e}")
            print(f"{Fore.CYAN}💡 Для использования vision установите: pip install opencv-python Pillow numpy")
            vision_handler = None
        
        # Инициализация конфигурации и модели
        config = LLMConfig(args.model)  # Используем конфиг по умолчанию
        
        # Показываем текущую конфигурацию
        config.print_config()
        
        # Проверка существования модели
        if not Path(config.model_path).exists():
            print(f"{Fore.RED}❌ Модель не найдена: {config.model_path}")
            print(f"{Fore.YELLOW}💡 Используйте параметр --model для указания правильного пути:")
            print(f"{Fore.CYAN}   python main.py --model \"C:/path/to/your/model.gguf\"")
            print(f"{Fore.YELLOW}💡 Или используйте --test для проверки зависимостей:")
            print(f"{Fore.CYAN}   python main.py --test")
            return
        
        llm = ThinkingLLM(config)
        
        print_separator()
        print(f"{Fore.GREEN}✓ Система готова к работе!")
        
        # Консольный режим
        print(f"{Fore.YELLOW}Команды:")
        print(f"{Fore.CYAN}  'quit' или 'exit' - выход")
        print(f"{Fore.CYAN}  'clear' - очистить историю диалога")
        print(f"{Fore.CYAN}  'history' - показать историю диалога")
        if audio_handler and audio_handler.is_speech_recognition_available():
            print(f"{Fore.CYAN}  'listen' - режим голосового ввода")
        if audio_handler:
            print(f"{Fore.CYAN}  'audio on/off' - включить/отключить аудио")
        if vision_handler and vision_handler.is_camera_available():
            print(f"{Fore.CYAN}  'vision start/stop' - запуск/остановка анализа камеры")
            print(f"{Fore.CYAN}  'look' - анализ текущего кадра")
            print(f"{Fore.CYAN}  'cameras' - показать доступные камеры")
            print(f"{Fore.CYAN}  'camera <id>' - переключиться на камеру по ID")
            print(f"{Fore.CYAN}  'refresh cameras' - обновить список камер")
        print_separator()
        
        while True:
            try:
                # Ввод пользователя
                user_input = input(f"\n{Fore.YELLOW}👤 Ваш вопрос (или 'listen' для голосового ввода): {Style.RESET_ALL}")
                
                if user_input.lower().strip() in ['quit', 'exit', 'выход']:
                    print(f"{Fore.BLUE}До свидания! 👋")
                    break
                
                if user_input.lower().strip() == 'clear':
                    llm.clear_history()
                    continue
                
                if user_input.lower().strip() == 'history':
                    print(f"\n{Fore.MAGENTA}📚 История диалога:")
                    for i, msg in enumerate(llm.conversation_history):
                        if msg['role'] == 'system':
                            continue
                        role_emoji = "👤" if msg['role'] == 'user' else "🤖"
                        content_preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                        print(f"{Fore.CYAN}{i}. {role_emoji} {msg['role']}: {content_preview}")
                    print(f"{Fore.BLUE}{llm.get_history_summary()}")
                    continue
                
                # Команды управления аудио
                if audio_handler and user_input.lower().strip() == 'audio on':
                    audio_handler.toggle_stt(True)
                    audio_handler.toggle_tts(True)
                    continue
                
                if audio_handler and user_input.lower().strip() == 'audio off':
                    audio_handler.toggle_stt(False)
                    audio_handler.toggle_tts(False)
                    continue
                
                # Голосовой ввод
                if user_input.lower().strip() == 'listen':
                    if audio_handler and audio_handler.is_speech_recognition_available():
                        print(f"{Fore.CYAN}🎤 Переход в режим голосового ввода...")
                        voice_input = audio_handler.listen_for_question()
                        if voice_input:
                            print(f"{Fore.GREEN}📝 Распознано: {voice_input}")
                            user_input = voice_input
                        else:
                            print(f"{Fore.YELLOW}❓ Голосовой ввод не распознан, попробуйте еще раз")
                            continue
                    else:
                        print(f"{Fore.RED}❌ Распознавание речи недоступно")
                        continue
                
                # Команды управления vision
                if vision_handler and user_input.lower().strip() == 'vision start':
                    if vision_handler.start_real_time_analysis():
                        print(f"{Fore.GREEN}📹 Анализ камеры запущен")
                    else:
                        print(f"{Fore.RED}❌ Не удалось запустить анализ камеры")
                    continue
                
                if vision_handler and user_input.lower().strip() == 'vision stop':
                    vision_handler.stop_real_time_analysis()
                    print(f"{Fore.YELLOW}⏹️  Анализ камеры остановлен")
                    continue
                
                # Команды управления камерами
                if vision_handler and user_input.lower().strip() == 'cameras':
                    available_cameras = vision_handler.list_cameras()
                    current_camera = vision_handler.get_current_camera()
                    print(f"{Fore.CYAN}📹 Текущая камера: {current_camera}")
                    if available_cameras:
                        print(f"{Fore.GREEN}✓ Доступные камеры: {available_cameras}")
                    else:
                        print(f"{Fore.RED}❌ Камеры не найдены")
                    continue
                
                if vision_handler and user_input.lower().strip().startswith('camera '):
                    try:
                        camera_id = int(user_input.split()[1])
                        if vision_handler.switch_camera(camera_id):
                            print(f"{Fore.GREEN}✓ Переключились на камеру {camera_id}")
                        else:
                            print(f"{Fore.RED}❌ Не удалось переключиться на камеру {camera_id}")
                    except (ValueError, IndexError):
                        print(f"{Fore.RED}❌ Неправильный формат. Используйте: camera <id>")
                    continue
                
                if vision_handler and user_input.lower().strip() == 'refresh cameras':
                    available_cameras = vision_handler.refresh_cameras()
                    if available_cameras:
                        print(f"{Fore.GREEN}✓ Обновлено! Доступные камеры: {available_cameras}")
                    else:
                        print(f"{Fore.RED}❌ Камеры не найдены")
                    continue
                
                if user_input.lower().strip() == 'look':
                    if vision_handler and vision_handler.is_camera_available():
                        print(f"{Fore.CYAN}👁️  Анализ текущего кадра...")
                        analysis = vision_handler.analyze_single_frame("Опишите что вы видите на изображении")
                        print(f"{Fore.GREEN}🔍 Анализ: {analysis}")
                        continue
                    else:
                        print(f"{Fore.RED}❌ Камера недоступна")
                        continue
                
                if not user_input.strip():
                    continue
                
                print_separator()
                
                # Генерация ответа
                response = llm.generate_response(user_input)
                
                # Вывод thinking процесса
                if response["thinking"]:
                    print_thinking(response["thinking"])
                
                # Вывод финального ответа
                if response["answer"]:
                    print_answer(response["answer"])
                else:
                    print(f"{Fore.RED}Ошибка: пустой ответ от модели")
                
                # Вывод статистики токенов
                if "token_stats" in response:
                    print_token_stats(response["token_stats"])
                
                # Озвучивание ответа
                if audio_handler and audio_handler.is_text_to_speech_available() and response["answer"]:
                    print(f"\n{Fore.CYAN}🔊 Озвучивание ответа...")
                    audio_handler.speak_answer(response["answer"], play_async=True)
                
                print_separator()
                
            except KeyboardInterrupt:
                print(f"\n{Fore.BLUE}Прервано пользователем. До свидания! 👋")
                break
            except Exception as e:
                print(f"{Fore.RED}Ошибка: {e}")
                continue
        
        # Очистка ресурсов
        if vision_handler:
            vision_handler.cleanup()
                
    except Exception as e:
        print(f"{Fore.RED}Критическая ошибка: {e}")
        sys.exit(1)
    finally:
        # Финальная очистка
        try:
            # Используем globals() вместо locals() для проверки переменной
            if 'vision_handler' in globals() and 'vision_handler' in locals():
                vision_handler_obj = locals().get('vision_handler')
                if vision_handler_obj:
                    vision_handler_obj.cleanup()
        except Exception:
            pass  # Игнорируем ошибки при очистке

if __name__ == "__main__":
    main()