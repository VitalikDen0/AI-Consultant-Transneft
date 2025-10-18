"""
Импорт готовой модели и анимаций в glTF для веба
"""

import bpy
import os

print("🚀 Импорт модели и анимаций...")

PROJECT_PATH = r"C:\Users\vital\Desktop\Hackathon"
MODEL_DIR = os.path.join(PROJECT_PATH, "3D")
OUTPUT_PATH = os.path.join(PROJECT_PATH, "static", "models", "transneft_character.glb")

# Очистка сцены
if bpy.context.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# ========== ИМПОРТ БАЗОВОЙ МОДЕЛИ С СКЕЛЕТОМ ==========

print("📦 Импорт базовой FBX модели со скелетом...")
base_fbx = os.path.join(MODEL_DIR, "Idle.fbx")  # Используем Idle как базу (со скелетом)

# Импорт первого FBX как основы
try:
    bpy.ops.wm.fbx_import(filepath=base_fbx)
    print("   ✅ Использован новый API импорта (Blender 4.x)")
except AttributeError:
    bpy.ops.import_scene.fbx(filepath=base_fbx)
    print("   ✅ Использован старый API импорта (Blender 3.x)")

# Получить армature и модель
armature = None
model = None

for obj in bpy.context.selected_objects:
    if obj.type == 'ARMATURE':
        armature = obj
        armature.name = "CharacterArmature"
    elif obj.type == 'MESH':
        model = obj
        model.name = "Character"

if not armature:
    print("❌ ОШИБКА: Армature не найден в FBX!")
    raise Exception("Армature не найден")

if not model:
    print("❌ ОШИБКА: Модель не найдена в FBX!")
    raise Exception("Модель не найдена")

print(f"   ✅ Армature: {armature.name}")
print(f"   ✅ Модель: {model.name}")

# Сохранить базовую анимацию Idle
if armature.animation_data and armature.animation_data.action:
    base_action = armature.animation_data.action
    base_action.name = "Idle"
    print(f"   ✅ Базовая анимация Idle сохранена")

# ========== ПРИМЕНЕНИЕ ТЕКСТУРЫ ИЗ OBJ ==========

print("🎨 Применение текстуры из OBJ...")
texture_png = os.path.join(MODEL_DIR, "Bot.png")

if os.path.exists(texture_png):
    # Загрузить текстуру
    texture_image = bpy.data.images.load(texture_png)
    print(f"   ✅ Загружена текстура: {os.path.basename(texture_png)}")
    
    # Создать материал с текстурой
    mat = bpy.data.materials.new(name="Character_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Очистить дефолтные ноды
    nodes.clear()
    
    # Создать Principled BSDF
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    
    # Создать нод текстуры
    tex_node = nodes.new('ShaderNodeTexImage')
    tex_node.image = texture_image
    tex_node.location = (-300, 0)
    
    # Создать Material Output
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    
    # Связать ноды
    links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    # Применить материал к модели
    if model.data.materials:
        model.data.materials[0] = mat
    else:
        model.data.materials.append(mat)
    
    print(f"   ✅ Материал применён к модели {model.name}")
else:
    print(f"   ⚠️ Текстура не найдена: {texture_png}")

# ========== ИМПОРТ ОСТАЛЬНЫХ АНИМАЦИЙ ==========

animations = {
    "Hello": "Hello.fbx",
    "Talking": "Talking.fbx",
    "Thinking": "Thinking.fbx"
}

print("\n🎬 Импорт остальных анимаций...")

for anim_name, fbx_file in animations.items():
    fbx_path = os.path.join(MODEL_DIR, fbx_file)
    
    if not os.path.exists(fbx_path):
        print(f"   ⚠️  Файл не найден: {fbx_file}")
        continue
    
    # Импорт FBX только для анимации
    try:
        bpy.ops.wm.fbx_import(filepath=fbx_path, use_anim=True)
    except AttributeError:
        bpy.ops.import_scene.fbx(filepath=fbx_path, use_anim=True)
    
    # Найти импортированный армature и его action
    imported_objects = list(bpy.context.selected_objects)
    imported_arm = None
    
    for obj in imported_objects:
        if obj.type == 'ARMATURE':
            imported_arm = obj
            break
    
    if imported_arm and imported_arm.animation_data and imported_arm.animation_data.action:
        # Взять анимацию и переименовать
        action = imported_arm.animation_data.action
        action.name = anim_name
        print(f"   ✅ {anim_name}: {len(action.fcurves)} fcurves")
    
    # Удалить все импортированные объекты (анимация уже сохранена в bpy.data.actions)
    for obj in list(imported_objects):
        try:
            if obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj)
        except ReferenceError:
            pass  # Объект уже удалён

# ========== ПРОВЕРКА ПРИВЯЗКИ ==========

print("\n🔗 Проверка привязки модели к скелету...")

if model.parent == armature:
    print("   ✅ Модель уже привязана к скелету")
else:
    print("   ⚠️ Модель не привязана, привязываем...")
    bpy.ops.object.select_all(action='DESELECT')
    model.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    print("   ✅ Привязка выполнена")

# ========== ЭКСПОРТ ==========

print("\n📤 Экспорт в glTF...")

# Проверка финальных анимаций
print("Проверка анимаций:")
for name in ["Hello", "Idle", "Talking", "Thinking"]:
    action = bpy.data.actions.get(name)
    if action:
        print(f"   ✅ {name} ({len(action.fcurves)} fcurves)")
    else:
        print(f"   ⚠️  {name} не найдена")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

bpy.ops.export_scene.gltf(
    filepath=OUTPUT_PATH,
    export_format='GLB',
    export_animations=True,
    use_selection=False
)

file_size = os.path.getsize(OUTPUT_PATH) / 1024
print(f"\n✅ Экспортировано: {OUTPUT_PATH}")
print(f"📦 Размер: {file_size:.1f} KB")
print("\n🎉 Готово! Запусти web_app.py и перезагрузи страницу")
