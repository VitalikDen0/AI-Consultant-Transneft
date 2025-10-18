/**
 * 3D Персонаж Транснефть для веб-интерфейса
 * Использует Three.js для рендеринга и анимации
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

class Character3D {
    constructor(container) {
        this.container = container;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.model = null;
        this.mixer = null;
        this.animations = {};
        this.currentAnimation = null;
        this.clock = new THREE.Clock();
        this.idleTimer = null;
        this.idleDelay = 5000; // 5 секунд
        
        this.init();
    }
    
    init() {
        console.log('🎭 Инициализация 3D персонажа...');
        
        // Создание сцены
        this.scene = new THREE.Scene();
        this.scene.background = null; // Прозрачный фон
        
        // Камера - оптимальная позиция для просмотра персонажа
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
        this.camera.position.set(0, 2, 5); // Камера выше (2) и дальше (5 единиц)
        this.camera.lookAt(0, 1.7, 0); // Смотрит чуть выше
        
        // Рендерер
        this.renderer = new THREE.WebGLRenderer({ 
            alpha: true, 
            antialias: true 
        });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.container.appendChild(this.renderer.domElement);
        
        // Освещение (увеличиваем яркость)
        const ambientLight = new THREE.AmbientLight(0xffffff, 1.0); // Было 0.6
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1.5); // Было 0.8
        directionalLight.position.set(5, 10, 7.5);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);
        
        const fillLight = new THREE.DirectionalLight(0x4A9EFF, 0.5); // Было 0.3
        fillLight.position.set(-5, 5, -5);
        this.scene.add(fillLight);
        
        // Добавляем подсветку снизу
        const bottomLight = new THREE.DirectionalLight(0xffffff, 0.3);
        bottomLight.position.set(0, -5, 0);
        this.scene.add(bottomLight);
        
        // Загрузка модели
        this.loadModel();
        
        // Обработка изменения размера
        window.addEventListener('resize', () => this.onResize());
        
        // Запуск анимационного цикла
        this.animate();
    }
    
    loadModel() {
        console.log('📦 Загрузка модели...');
        
        const loader = new GLTFLoader();
        
        loader.load(
            '/static/models/transneft_character.glb',
            (gltf) => {
                console.log('✅ Модель загружена');
                this.model = gltf.scene;
                
                // Вычисляем размер модели
                const box = new THREE.Box3().setFromObject(this.model);
                const size = box.getSize(new THREE.Vector3());
                const center = box.getCenter(new THREE.Vector3());
                
                console.log('📏 Размер модели:', size);
                console.log('📍 Центр модели:', center);
                
                // FBX модель из Mixamo уже в правильной ориентации - не поворачиваем!
                // (OBJ требовал поворота, FBX - нет)
                
                // Нормализуем размер (делаем модель высотой ~2 единицы)
                const maxSize = Math.max(size.x, size.y, size.z);
                const scale = 2 / maxSize;
                this.model.scale.set(scale, scale, scale);
                
                // Позиционируем модель:
                // - По X и Z центрируем
                // - По Y ставим на сетку (основание модели на Y=0)
                this.model.position.set(
                    -center.x * scale,  // Центр по X
                    0,                   // Ставим на уровень земли
                    -center.z * scale   // Центр по Z
                );
                
                console.log('📐 Применен масштаб:', scale);
                console.log('📍 Финальная позиция:', this.model.position);
                
                // Проверка и восстановление материалов/текстур
                this.model.traverse((child) => {
                    if (child.isMesh) {
                        console.log('🎨 Mesh найден:', child.name);
                        if (child.material) {
                            console.log('   Материал:', child.material.type);
                            // Включаем текстуры если есть
                            if (child.material.map) {
                                child.material.map.needsUpdate = true;
                                console.log('   ✅ Текстура активна');
                            } else {
                                console.log('   ⚠️ Текстура отсутствует');
                            }
                            child.material.needsUpdate = true;
                        }
                    }
                });
                
                this.scene.add(this.model);
                
                // Настройка анимаций
                if (gltf.animations && gltf.animations.length > 0) {
                    this.mixer = new THREE.AnimationMixer(this.model);
                    
                    console.log(`✅ Найдено анимаций: ${gltf.animations.length}`);
                    
                    gltf.animations.forEach((clip) => {
                        console.log(`🎬 Найдена анимация: ${clip.name}`);
                        this.animations[clip.name] = this.mixer.clipAction(clip);
                    });
                    
                    // Автоматический запуск приветствия
                    this.greet();
                } else {
                    console.warn('⚠️  Анимации не найдены в модели');
                }
                
                console.log('🎉 Персонаж готов!');
            },
            (progress) => {
                const percent = (progress.loaded / progress.total * 100).toFixed(0);
                console.log(`📊 Загрузка: ${percent}%`);
            },
            (error) => {
                console.error('❌ Ошибка загрузки модели:', error);
            }
        );
    }
    
    stopAllAnimations() {
        // Останавливаем ВСЕ анимации и возвращаем в T-pose
        if (this.mixer) {
            Object.values(this.animations).forEach(action => {
                action.stop();
                action.weight = 0;
            });
            // Сбрасываем mixer в начальное состояние
            this.mixer.stopAllAction();
        }
    }
    
    playAnimation(name, loop = false) {
        console.log(`▶️  Воспроизведение: ${name} (loop: ${loop})`);
        
        if (!this.mixer || !this.animations[name]) {
            console.warn(`⚠️  Анимация '${name}' не найдена`);
            console.log('Доступные анимации:', Object.keys(this.animations));
            return;
        }
        
        // ПОЛНАЯ остановка всех анимаций и возврат в T-pose
        this.stopAllAnimations();
        
        // Запуск новой анимации с чистого листа
        const action = this.animations[name];
        action.reset(); // Сброс в начальное положение
        action.time = 0; // Время на 0
        action.weight = 1; // Полный вес
        action.setEffectiveWeight(1);
        action.setLoop(loop ? THREE.LoopRepeat : THREE.LoopOnce);
        action.clampWhenFinished = true;
        action.play();
        
        this.currentAnimation = action;
        
        // Событие окончания анимации
        if (!loop) {
            this.mixer.addEventListener('finished', (e) => {
                if (e.action === action) {
                    console.log(`✓ Анимация '${name}' завершена`);
                    this.onAnimationFinished(name);
                }
            });
        }
    }
    
    onAnimationFinished(name) {
        // Переход в idle после завершения анимации
        console.log('💤 Возврат в idle состояние');
        this.currentAnimation = null;
        this.resetIdleTimer();
    }
    
    // Методы для управления из приложения
    startThinking() {
        console.log('🤔 Состояние: Думает');
        this.resetIdleTimer();
        this.playAnimation('Thinking', true);
    }
    
    startAnswering() {
        console.log('💬 Состояние: Отвечает');
        this.resetIdleTimer();
        this.playAnimation('Talking', true);
    }
    
    stopAnimation() {
        console.log('⏸️  Остановка анимации - доигрываем до конца');
        
        // Если сейчас играет Talking, переключаем его в режим "один раз"
        if (this.currentAnimation && this.animations['Talking'] === this.currentAnimation) {
            // Переключаем loop на false - анимация доиграет до конца
            this.currentAnimation.setLoop(THREE.LoopOnce);
            this.currentAnimation.clampWhenFinished = true;
            
            // Подписываемся на событие завершения
            this.mixer.addEventListener('finished', (e) => {
                if (e.action === this.currentAnimation) {
                    console.log('✓ Talking завершён, переход в Idle');
                    this.resetIdleTimer(); // Запустить таймер для перехода в Idle
                }
            }, { once: true }); // Один раз
        } else {
            // Для других анимаций - обычная остановка
            if (this.currentAnimation) {
                this.currentAnimation.fadeOut(0.5);
                this.currentAnimation = null;
            }
            this.resetIdleTimer();
        }
    }
    
    resetIdleTimer() {
        // Сбросить таймер
        if (this.idleTimer) {
            clearTimeout(this.idleTimer);
        }
        
        // Запустить новый таймер - через 5 сек показать Idle анимацию
        this.idleTimer = setTimeout(() => {
            console.log('😴 Простой - переход в Idle');
            this.playAnimation('Idle', true);
        }, this.idleDelay);
    }
    
    greet() {
        console.log('👋 Приветствие');
        this.playAnimation('Hello', false);
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        const delta = this.clock.getDelta();
        
        // Обновление анимаций
        if (this.mixer) {
            this.mixer.update(delta);
        }
        
        // Лёгкое покачивание модели для живости
        if (this.model && !this.currentAnimation) {
            const time = this.clock.getElapsedTime();
            this.model.rotation.y = Math.sin(time * 0.3) * 0.05;
            this.model.position.y = -0.5 + Math.sin(time * 0.5) * 0.02;
        }
        
        this.renderer.render(this.scene, this.camera);
    }
    
    onResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        
        this.renderer.setSize(width, height);
    }
    
    dispose() {
        console.log('🗑️  Очистка 3D ресурсов');
        
        if (this.mixer) {
            this.mixer.stopAllAction();
        }
        
        if (this.renderer) {
            this.renderer.dispose();
        }
        
        if (this.scene) {
            this.scene.traverse((object) => {
                if (object.geometry) {
                    object.geometry.dispose();
                }
                if (object.material) {
                    if (Array.isArray(object.material)) {
                        object.material.forEach(material => material.dispose());
                    } else {
                        object.material.dispose();
                    }
                }
            });
        }
    }
}

// Экспорт для использования в приложении
export { Character3D };
