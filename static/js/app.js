// AI-Консультант Транснефть - Web Application

import { Character3D } from './character3d.js';

class TransneftAssistant {
    constructor() {
        this.messagesArea = document.getElementById('messagesArea');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.micButton = document.getElementById('micButton');
        this.cameraButton = document.getElementById('cameraButton');
        this.speakerButton = document.getElementById('speakerButton');
        this.thinkingPanel = document.getElementById('thinkingPanel');
        this.thinkingContent = document.getElementById('thinkingContent');
        this.statusText = document.getElementById('statusText');
        this.voiceIndicator = document.getElementById('voiceIndicator');
        this.cameraIndicator = document.getElementById('cameraIndicator');
        
        this.isVoiceActive = false;
        this.isCameraActive = false;
        this.isSpeakerActive = false; // Озвучка ответов включена/выключена
        this.isProcessing = false;
        this.voicePollingInterval = null; // Интервал для опроса распознанного текста
        this.gesturePollingInterval = null; // Интервал для опроса жестов
        
        // 3D Персонаж
        this.character3d = null;
        
        this.init();
    }
    
    async init() {
        // Сброс контекста при загрузке страницы
        await this.resetContext();
        
        // Инициализация 3D персонажа
        this.init3DCharacter();
        
        // Обработчики событий
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        this.micButton.addEventListener('click', () => this.toggleVoice());
        this.cameraButton.addEventListener('click', () => this.toggleCamera());
        this.speakerButton.addEventListener('click', () => this.toggleSpeaker());
        
        // Авто-ресайз textarea
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
        });
    }
    
    init3DCharacter() {
        console.log('🎭 Инициализация 3D персонажа...');
        const container = document.getElementById('character3dContainer');
        
        if (!container) {
            console.warn('⚠️  Контейнер для 3D не найден');
            return;
        }
        
        if (typeof Character3D === 'undefined') {
            console.error('❌ Character3D класс не загружен');
            return;
        }
        
        try {
            this.character3d = new Character3D(container);
            
            // Приветствие через 1 секунду после загрузки
            setTimeout(() => {
                if (this.character3d) {
                    this.character3d.greet();
                    this.updateStatus('Приветствую! 👋');
                    setTimeout(() => this.updateStatus('Готов к работе'), 2000);
                }
            }, 1000);
        } catch (error) {
            console.error('❌ Ошибка инициализации 3D персонажа:', error);
        }
    }
    
    updateStatus(text) {
        this.statusText.textContent = text;
    }
    
    async sendMessage(source = 'text') {
        const message = this.messageInput.value.trim();
        if (!message || this.isProcessing) return;
        
        // Добавляем сообщение пользователя
        this.addMessage(message, true);
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        
        // Блокируем ввод
        this.setProcessing(true);
        this.setStatus('thinking', 'Размышляю...');
        
        // Запускаем анимацию печати на планшете
        if (this.character3d) {
            this.character3d.startThinking();
        }
        
        try {
            // Fetch для streaming ответов
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    message: message,
                    source: source  // text/voice/gesture
                })
            });
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let lastAnswerBubble = null;
            let fullAnswerText = ''; // Накапливаем полный ответ для озвучки
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n\n');
                buffer = lines.pop() || '';
                
                for (const line of lines) {
                    if (!line.trim() || !line.startsWith('data: ')) continue;
                    
                    try {
                        const data = JSON.parse(line.slice(6));
                        
                        if (data.type === 'thinking') {
                            // Показываем thinking
                            this.showThinking(data.content);
                        } else if (data.type === 'answer') {
                            // Обновляем или создаем ответ
                            if (!lastAnswerBubble) {
                                lastAnswerBubble = this.addMessage('', false, true);
                                this.hideThinking();
                                this.setStatus('speaking', 'Отвечаю...');
                                
                                // Переключаем на анимацию жестикуляции
                                if (this.character3d) {
                                    this.character3d.startAnswering();
                                }
                            }
                            fullAnswerText = data.content; // Сохраняем полный текст
                            this.updateMessage(lastAnswerBubble, data.content);
                        } else if (data.type === 'done') {
                            // Завершение
                            this.setStatus('idle', 'Готов к работе');
                            this.setProcessing(false);
                            
                            // Останавливаем анимацию, возврат к idle
                            if (this.character3d) {
                                this.character3d.stopAnimation();
                            }
                            
                            // Озвучиваем финальный ответ если включена автоозвучка
                            if (fullAnswerText && this.isSpeakerActive) {
                                await this.speakText(fullAnswerText);
                            }
                            
                            // Возобновляем голосовой ввод и распознавание жестов если были активны
                            await this.resumeVoiceAfterAnswer();
                            await this.resumeGestureAfterAnswer();
                        }
                    } catch (e) {
                        console.error('Ошибка парсинга:', e);
                    }
                }
            }
        } catch (error) {
            console.error('Ошибка:', error);
            this.addMessage('Извините, произошла ошибка. Попробуйте ещё раз.', false);
            this.setStatus('idle', 'Готов к работе');
            this.setProcessing(false);
            
            // Останавливаем анимацию при ошибке
            if (this.character3d) {
                this.character3d.stopAnimation();
            }
        }
    }
    
    addMessage(text, isUser = false, isStreaming = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
        
        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'message-bubble';
        
        if (!isUser) {
            // Заголовок с аватаром
            const headerDiv = document.createElement('div');
            headerDiv.className = 'message-header';
            headerDiv.innerHTML = `
                <span class="avatar">👨‍💼</span>
                <span class="name">AI-Консультант</span>
            `;
            bubbleDiv.appendChild(headerDiv);
        }
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        if (isStreaming) {
            // Добавляем typing индикатор
            contentDiv.innerHTML = `
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            `;
        } else {
            // Форматируем текст с поддержкой Markdown
            const paragraphs = text.split('\n').filter(p => p.trim());
            contentDiv.innerHTML = paragraphs.map(p => `<p>${this.formatMarkdown(p)}</p>`).join('');
        }
        
        bubbleDiv.appendChild(contentDiv);
        messageDiv.appendChild(bubbleDiv);
        this.messagesArea.appendChild(messageDiv);
        
        // Скроллим вниз
        this.scrollToBottom();
        
        return contentDiv;
    }
    
    updateMessage(contentDiv, text) {
        // Убираем typing индикатор если есть
        const typingIndicator = contentDiv.querySelector('.typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
        
        // Обновляем текст с поддержкой Markdown
        const paragraphs = text.split('\n').filter(p => p.trim());
        contentDiv.innerHTML = paragraphs.map(p => `<p>${this.formatMarkdown(p)}</p>`).join('');
        
        // Скроллим вниз
        this.scrollToBottom();
    }
    
    showThinking(content) {
        this.thinkingPanel.style.display = 'block';
        this.thinkingContent.textContent = '';
        
        // Анимация посимвольного появления
        let index = 0;
        const interval = setInterval(() => {
            if (index < content.length) {
                this.thinkingContent.textContent += content[index];
                index++;
            } else {
                clearInterval(interval);
            }
        }, 10);
    }
    
    hideThinking() {
        // Плавное исчезновение
        this.thinkingPanel.style.transition = 'opacity 0.5s ease-out';
        this.thinkingPanel.style.opacity = '0';
        
        setTimeout(() => {
            this.thinkingPanel.style.display = 'none';
            this.thinkingPanel.style.opacity = '1';
        }, 500);
    }
    
    setStatus(state, text) {
        if (this.statusText) {
            this.statusText.textContent = text;
        }
        
        // Эмодзи больше не используется, только 3D персонаж
    }
    
    setProcessing(isProcessing) {
        this.isProcessing = isProcessing;
        this.messageInput.disabled = isProcessing;
        this.sendButton.disabled = isProcessing;
        
        if (isProcessing) {
            this.sendButton.style.opacity = '0.6';
            this.sendButton.style.cursor = 'not-allowed';
        } else {
            this.sendButton.style.opacity = '1';
            this.sendButton.style.cursor = 'pointer';
        }
    }
    
    async toggleVoice() {
        this.isVoiceActive = !this.isVoiceActive;
        
        if (this.isVoiceActive) {
            this.micButton.classList.add('active');
            this.voiceIndicator.classList.add('active');
            this.setStatus('listening', 'Слушаю... (2 сек молчания → отправка, 5 сек → выкл)');
            
            try {
                const response = await fetch('/api/voice/start', { method: 'POST' });
                const data = await response.json();
                
                if (data.status === 'started') {
                    console.log('🎤 Голосовой ввод активирован');
                    // Начинаем опрос для получения распознанного текста
                    this.startVoicePolling();
                } else {
                    console.error('Ошибка старта:', data);
                    this.isVoiceActive = false;
                    this.micButton.classList.remove('active');
                    this.voiceIndicator.classList.remove('active');
                }
            } catch (error) {
                console.error('Ошибка запуска голоса:', error);
                this.isVoiceActive = false;
                this.micButton.classList.remove('active');
                this.voiceIndicator.classList.remove('active');
            }
        } else {
            this.micButton.classList.remove('active');
            this.voiceIndicator.classList.remove('active');
            this.setStatus('idle', 'Готов к работе');
            this.stopVoicePolling();
            
            try {
                await fetch('/api/voice/stop', { method: 'POST' });
            } catch (error) {
                console.error('Ошибка остановки голоса:', error);
            }
        }
    }
    
    startVoicePolling() {
        // Опрашиваем сервер каждые 500ms для получения распознанного текста
        this.voicePollingInterval = setInterval(async () => {
            try {
                const response = await fetch('/api/voice/get-text');
                const data = await response.json();
                
                if (data.status === 'text_available' && data.text) {
                    console.log('🗣️ Получен текст:', data.text);
                    
                    // ОСТАНАВЛИВАЕМ POLLING НА ВРЕМЯ ОБРАБОТКИ ОТВЕТА
                    this.stopVoicePolling();
                    
                    // Добавляем текст в поле ввода
                    this.messageInput.value = data.text;
                    
                    // Автоматически отправляем нейросети
                    await this.sendMessage('voice');
                    
                    // После ответа нейросети возобновляем запись (см. sendMessage)
                } else if (data.status === 'stopped') {
                    // Запись остановлена (таймаут 5 секунд) - прекращаем polling
                    console.log('⏰ Таймаут - запись остановлена, возврат кнопки в исходное состояние');
                    this.stopVoicePolling();
                    
                    // Полностью сбрасываем состояние голосового ввода
                    this.isVoiceActive = false;
                    this.micButton.classList.remove('active');
                    this.voiceIndicator.classList.remove('active');
                    
                    // Показываем уведомление пользователю
                    this.showNotification('Голосовой ввод остановлен (таймаут 5 сек)', 'warning');
                }
            } catch (error) {
                console.error('Ошибка получения текста:', error);
            }
        }, 500);
    }
    
    stopVoicePolling() {
        if (this.voicePollingInterval) {
            clearInterval(this.voicePollingInterval);
            this.voicePollingInterval = null;
        }
    }
    
    startGesturePolling() {
        // Опрашиваем сервер каждые 500ms для получения распознанных жестов
        this.gesturePollingInterval = setInterval(async () => {
            try {
                const response = await fetch('/api/gesture/get-text');
                const data = await response.json();
                
                if (data.status === 'text_available' && data.text) {
                    console.log('👋 Получен жест:', data.text);
                    
                    // ОСТАНАВЛИВАЕМ POLLING НА ВРЕМЯ ОБРАБОТКИ ОТВЕТА
                    this.stopGesturePolling();
                    
                    // Добавляем текст в поле ввода
                    this.messageInput.value = data.text;
                    
                    // Автоматически отправляем нейросети
                    await this.sendMessage('gesture');
                    
                    // После ответа нейросети камера автоматически возобновится (в web_app.py)
                } else if (data.status === 'stopped') {
                    // Камера остановлена (таймаут 7 секунд) - прекращаем polling
                    console.log('⏰ Таймаут - камера остановлена, возврат кнопки в исходное состояние');
                    this.stopGesturePolling();
                    
                    // Полностью сбрасываем состояние камеры
                    this.isCameraActive = false;
                    this.cameraButton.classList.remove('active');
                    this.cameraIndicator.classList.remove('active');
                    
                    // Показываем уведомление пользователю
                    this.showNotification('Распознавание жестов остановлено (таймаут 7 сек)', 'warning');
                }
            } catch (error) {
                console.error('Ошибка получения жеста:', error);
            }
        }, 500);
    }
    
    stopGesturePolling() {
        if (this.gesturePollingInterval) {
            clearInterval(this.gesturePollingInterval);
            this.gesturePollingInterval = null;
        }
    }
    
    async resumeGestureAfterAnswer() {
        // Возобновляем распознавание жестов после ответа нейросети
        if (this.isCameraActive && this.cameraButton.classList.contains('active')) {
            console.log('📹 Возобновляем опрос жестов после ответа');
            // Камера автоматически возобновлена на backend (web_app.py)
            // Просто продолжаем опрос
            this.startGesturePolling();
        }
    }
    
    async resumeVoiceAfterAnswer() {
        // Возобновляем голосовой ввод после ответа нейросети ТОЛЬКО если кнопка активна
        if (this.isVoiceActive && this.micButton.classList.contains('active')) {
            try {
                const response = await fetch('/api/voice/resume', { method: 'POST' });
                const data = await response.json();
                
                if (data.status === 'started' || data.status === 'already_listening') {
                    console.log('🎤 Голосовой ввод возобновлён после ответа');
                    // ВОЗОБНОВЛЯЕМ POLLING после ответа
                    this.startVoicePolling();
                } else {
                    console.log('⚠️ Не удалось возобновить:', data);
                    // Если не удалось возобновить - сбрасываем UI
                    this.isVoiceActive = false;
                    this.micButton.classList.remove('active');
                    this.voiceIndicator.classList.remove('active');
                }
            } catch (error) {
                console.error('Ошибка возобновления голоса:', error);
                // При ошибке также сбрасываем UI
                this.isVoiceActive = false;
                this.micButton.classList.remove('active');
                this.voiceIndicator.classList.remove('active');
            }
        } else {
            console.log('ℹ️ Голосовой ввод неактивен - пропуск возобновления');
        }
    }
    
    toggleSpeaker() {
        this.isSpeakerActive = !this.isSpeakerActive;
        
        if (this.isSpeakerActive) {
            this.speakerButton.classList.add('active');
            console.log('🔊 Автоозвучка включена');
        } else {
            this.speakerButton.classList.remove('active');
            console.log('🔇 Автоозвучка выключена');
        }
    }
    
    async speakText(text) {
        /**
         * Озвучивание текста через TTS
         * Автоматически очищает эмодзи и лишние символы
         */
        if (!this.isSpeakerActive || !text || !text.trim()) {
            return;
        }
        
        try {
            console.log('🔊 Отправка на озвучку:', text.substring(0, 50) + '...');
            
            const response = await fetch('/api/tts/speak', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });
            
            const data = await response.json();
            
            if (data.status === 'started') {
                console.log('✅ Озвучка запущена');
            } else if (data.status === 'skipped') {
                console.log('⏭️ Озвучка пропущена:', data.message);
            } else {
                console.warn('⚠️ Озвучка:', data);
            }
        } catch (error) {
            console.error('❌ Ошибка озвучки:', error);
        }
    }
    
    async toggleCamera() {
        this.isCameraActive = !this.isCameraActive;
        
        if (this.isCameraActive) {
            this.cameraButton.classList.add('active');
            this.cameraIndicator.classList.add('active');
            this.setStatus('camera', 'Камера активна - распознавание жестов (7 сек → выкл)');
            
            try {
                const response = await fetch('/api/camera/start', { method: 'POST' });
                const data = await response.json();
                
                if (data.status === 'started') {
                    console.log('📹 Камера и распознавание жестов активированы');
                    // Начинаем опрос для получения распознанных жестов
                    this.startGesturePolling();
                } else {
                    console.error('Ошибка старта камеры:', data);
                    this.isCameraActive = false;
                    this.cameraButton.classList.remove('active');
                    this.cameraIndicator.classList.remove('active');
                }
            } catch (error) {
                console.error('Ошибка запуска камеры:', error);
                this.isCameraActive = false;
                this.cameraButton.classList.remove('active');
                this.cameraIndicator.classList.remove('active');
            }
        } else {
            this.cameraButton.classList.remove('active');
            this.cameraIndicator.classList.remove('active');
            this.setStatus('idle', 'Готов к работе');
            this.stopGesturePolling();
            
            try {
                await fetch('/api/camera/stop', { method: 'POST' });
            } catch (error) {
                console.error('Ошибка остановки камеры:', error);
            }
        }
    }
    
    scrollToBottom() {
        this.messagesArea.scrollTop = this.messagesArea.scrollHeight;
    }
    
    showNotification(message, type = 'info') {
        /**
         * Показывает временное уведомление пользователю
         * type: 'info', 'success', 'warning', 'error'
         */
        console.log(`📢 [${type.toUpperCase()}] ${message}`);
        
        // Используем статусную строку для показа уведомления
        const originalStatus = this.statusText.textContent;
        const icons = {
            'info': 'ℹ️',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌'
        };
        
        this.setStatus('idle', `${icons[type] || 'ℹ️'} ${message}`);
        
        // Возвращаем исходный статус через 3 секунды
        setTimeout(() => {
            if (!this.isProcessing) {
                this.setStatus('idle', 'Готов к работе');
            }
        }, 3000);
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    formatMarkdown(text) {
        /**
         * Конвертирует Markdown в HTML
         * Поддерживает:
         * - **жирный** → <strong>жирный</strong>
         * - *курсив* → <em>курсив</em>
         * - ***жирный курсив*** → <strong><em>жирный курсив</em></strong>
         * - `код` → <code>код</code>
         * - ```код блок``` → <pre><code>код блок</code></pre>
         * - # Заголовок → <h3>Заголовок</h3>
         * - ## Подзаголовок → <h4>Подзаголовок</h4>
         * - [ссылка](url) → <a href="url">ссылка</a>
         * - > цитата → <blockquote>цитата</blockquote>
         * - - список → <ul><li>список</li></ul>
         * - 1. список → <ol><li>список</li></ol>
         */
        
        // Сначала экранируем HTML для безопасности
        let html = this.escapeHtml(text);
        
        // Блоки кода (должны быть до инлайн кода)
        html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
        
        // Жирный курсив ***текст***
        html = html.replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>');
        
        // Жирный **текст**
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Курсив *текст* или _текст_
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
        html = html.replace(/_(.*?)_/g, '<em>$1</em>');
        
        // Инлайн код `текст`
        html = html.replace(/`(.*?)`/g, '<code>$1</code>');
        
        // Зачеркнутый ~~текст~~
        html = html.replace(/~~(.*?)~~/g, '<del>$1</del>');
        
        // Ссылки [текст](url)
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
        
        // Заголовки
        html = html.replace(/^### (.*?)$/gm, '<h5>$1</h5>');
        html = html.replace(/^## (.*?)$/gm, '<h4>$1</h4>');
        html = html.replace(/^# (.*?)$/gm, '<h3>$1</h3>');
        
        // Цитаты
        html = html.replace(/^&gt; (.*?)$/gm, '<blockquote>$1</blockquote>');
        
        // Горизонтальная линия
        html = html.replace(/^---$/gm, '<hr>');
        html = html.replace(/^\*\*\*$/gm, '<hr>');
        
        return html;
    }
    
    async resetContext() {
        // Сброс контекста нейросети при загрузке страницы
        try {
            const response = await fetch('/api/reset', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (response.ok) {
                console.log('✅ Контекст успешно сброшен');
            } else {
                console.error('❌ Ошибка сброса контекста:', await response.text());
            }
        } catch (error) {
            console.error('❌ Ошибка при сбросе контекста:', error);
        }
    }
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    const assistant = new TransneftAssistant();
    console.log('🚀 AI-Консультант Транснефть загружен');
});
