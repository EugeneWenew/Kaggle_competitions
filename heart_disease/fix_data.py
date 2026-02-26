import re
import time

def fix_train_csv(input_file='train.csv', output_file='train_fixed.csv'):
    """
    Исправляет CSV файл — добавляет переносы строк между записями
    """
    print("=" * 60)
    print("🔧 ИСПРАВЛЕНИЕ ФАЙЛА train.csv")
    print("=" * 60)
    
    start_time = time.time()
    
    # Читаем исходный файл
    print(f"\n📖 Чтение {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"✅ Прочитано {len(content):,} символов")
    
    # Добавляем переносы строк между записями
    # Паттерн: Absence или Presence, за которыми следует число (ID следующей записи)
    print("\n✏️ Добавление переносов строк...")
    fixed_content = re.sub(r'(Absence|Presence)(\d+)', r'\1\n\2', content)
    
    # Добавляем заголовок, если его нет
    lines = [line.strip() for line in fixed_content.strip().split('\n') if line.strip()]
    
    if not lines[0].startswith('id,'):
        print("📝 Добавление заголовка...")
        header = 'id,Age,Sex,Chest_pain_type,BP,Cholesterol,FBS_over_120,EKG_results,Max_HR,Exercise_angina,ST_depression,Slope_of_ST,Vessels_fluro,Thallium,Heart_Disease'
        lines = [header] + lines
    
    # Записываем исправленный файл
    print(f"\n💾 Запись в {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    # Считаем количество строк
    with open(output_file, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("✅ ГОТОВО!")
    print("=" * 60)
    print(f"⏱️ Время: {elapsed:.1f} секунд")
    print(f"📊 Строк в файле: {line_count:,}")
    print(f"📁 Файл сохранён: {output_file}")
    print("=" * 60)
    
    return line_count

# Запуск
if __name__ == "__main__":
    fix_train_csv()