"""
QA Benchmark - Бенчмарк по файлу gobench.md
Запускает тестирование LLM на 17 вопросах о Транснефть
"""

import re
from datetime import datetime
from pathlib import Path
import sys

# Добавляем путь к main.py
sys.path.insert(0, str(Path(__file__).parent))

from main import ThinkingLLM, LLMConfig


class QABenchmark:
    """Класс для запуска бенчмарка по gobench.md"""
    
    def __init__(self, questions_file: str = "gobench.md", log_file: str = None):
        self.questions_file = questions_file
        
        # Генерация имени лог-файла с датой и временем
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = f"qa_benchmark_{timestamp}.log"
        else:
            self.log_file = log_file
        
        # Инициализация LLM (используется PROMPT.md автоматически)
        print("🤖 Инициализация LLM...")
        config = LLMConfig()
        self.llm = ThinkingLLM(config)
        
        # Требования к ответам (будут добавлены к каждому вопросу)
        self.requirements = """
**Требования к ответам:**
- Максимальная детализация, точные ссылки на год, нормативный документ, цифры (километры, мегатонны, количество, адреса и реквизиты)
- Для вопросов расчётных — вывести формулы и рассуждения
- Если ответ невозможен — чётко указать, что информации нет (где нет — ясно обозначить пропуск)
"""
        
        print(f"📝 Файл вопросов: {self.questions_file}")
        print(f"📄 Лог-файл: {self.log_file}")
    
    def parse_questions(self) -> list[dict]:
        """Парсинг вопросов из gobench.md"""
        try:
            with open(self.questions_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Улучшенный паттерн для парсинга вопросов
            # Находим все вопросы (начинаются с **номер.**)
            pattern = r'\*\*(\d+)\.\s+([^\*]+(?:\n(?!\*\*\d+\.)[^\*]*)*)'
            matches = re.finditer(pattern, content, re.MULTILINE)
            
            questions = []
            for match in matches:
                question_num = match.group(1)
                question_text = match.group(2).strip()
                
                # Формируем полный вопрос с номером
                full_question = f"**Вопрос {question_num}:**\n{question_text}\n\n{self.requirements}"
                questions.append({
                    'number': question_num,
                    'text': question_text,
                    'full': full_question
                })
            
            # Валидация: должно быть ровно 17 вопросов
            if len(questions) != 17:
                print(f"⚠️ ВНИМАНИЕ: Найдено {len(questions)} вопросов вместо ожидаемых 17!")
                print(f"Номера найденных вопросов: {[q['number'] for q in questions]}")
                
                # Проверяем какие вопросы отсутствуют
                found_nums = set(int(q['number']) for q in questions)
                expected_nums = set(range(1, 18))
                missing = expected_nums - found_nums
                
                if missing:
                    print(f"❌ Отсутствующие вопросы: {sorted(missing)}")
                    print("\nПроверьте форматирование файла gobench.md!")
                    print("Каждый вопрос должен начинаться с **номер.** на новой строке")
            
            return questions
        
        except FileNotFoundError:
            print(f"❌ Файл {self.questions_file} не найден!")
            return []
        except Exception as e:
            print(f"❌ Ошибка парсинга: {e}")
            return []
    
    def log_message(self, message: str, console: bool = True):
        """Логирование сообщения в файл и консоль"""
        if console:
            print(message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    
    def run_benchmark(self):
        """Запуск бенчмарка"""
        # Парсинг вопросов
        questions = self.parse_questions()
        
        if not questions:
            print("❌ Не удалось извлечь вопросы из файла")
            return
        
        print(f"\n✅ Найдено {len(questions)} вопросов\n")
        
        # Заголовок в лог-файле
        separator = "=" * 80
        header = f"""
{separator}
QA BENCHMARK - Тестирование LLM по вопросам о ПАО «Транснефть»
Дата запуска: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Модель: {self.llm.config.model_path}
Количество вопросов: {len(questions)}
{separator}

"""
        self.log_message(header, console=False)
        
        # Обработка каждого вопроса
        for i, q in enumerate(questions, 1):
            question_header = f"\n{'=' * 80}\n📋 ВОПРОС {q['number']}/{len(questions)}\n{'=' * 80}\n"
            self.log_message(question_header)
            
            # Логируем текст вопроса
            self.log_message(f"\n{q['text']}\n")
            self.log_message(f"{'─' * 80}\n")
            
            # Генерация ответа
            print(f"🤔 Генерация ответа на вопрос {q['number']}...")
            try:
                response = self.llm.generate_response(q['full'])
                
                # Логируем thinking (если есть)
                thinking = response.get('thinking', '')
                if thinking:
                    self.log_message("💭 РАЗМЫШЛЕНИЯ (Thinking):\n", console=False)
                    self.log_message(thinking + "\n", console=False)
                    self.log_message(f"{'─' * 80}\n", console=False)
                
                # Логируем ответ
                answer = response.get('answer', '')
                self.log_message("✅ ОТВЕТ:\n")
                self.log_message(answer + "\n")
                
                print(f"✅ Вопрос {q['number']} обработан\n")
                
            except Exception as e:
                error_msg = f"❌ Ошибка при генерации ответа: {e}\n"
                self.log_message(error_msg)
                print(error_msg)
        
        # Финальное сообщение
        footer = f"""
{separator}
БЕНЧМАРК ЗАВЕРШЁН
Время завершения: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Результаты сохранены в: {self.log_file}
{separator}
"""
        self.log_message(footer)
        
        print(f"\n🎉 Бенчмарк завершён! Результаты в файле: {self.log_file}")


def main():
    """Точка входа"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║          QA BENCHMARK - ПАО «Транснефть»                     ║
║          Тестирование LLM на 17 вопросах                     ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Создание и запуск бенчмарка
    benchmark = QABenchmark()
    benchmark.run_benchmark()


if __name__ == "__main__":
    main()
