import json
from pathlib import Path

def validate_and_fix_project_json():
    
    json_file = Path('data/project.json')
    
    if not json_file.exists():
        print("❌ Файл data/project.json не найден!")
        return
    
    print("🔍 Проверка project.json...")
    
    # Загрузка и валидация JSON
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ ОШИБКА: Неверный JSON формат!")
        print(f"   Строка {e.lineno}, позиция {e.colno}")
        print(f"   {e.msg}")
        return
    
    if not isinstance(data, list):
        print("❌ ОШИБКА: project.json должен содержать массив объектов")
        return
    
    print(f"✅ JSON синтаксис корректен")
    print(f"📊 Найдено записей: {len(data)}")
    
    # Статистика
    stats = {
        'total': len(data),
        'valid': 0,
        'missing_text': 0,
        'missing_labels': 0,
        'duplicates': 0,
        'fixed': []
    }
    
    seen_texts = set()
    cleaned_data = []
    
    for idx, item in enumerate(data):
        issues = []
        
        # Извлечение текста (разные форматы)
        text = None
        if 'text' in item:
            text = item['text']
        elif 'data' in item and isinstance(item['data'], dict):
            text = item['data'].get('text')
        
        # Проверка наличия текста
        if not text or not text.strip():
            stats['missing_text'] += 1
            issues.append(f"  ⚠️ Запись #{idx}: Пустой текст")
            continue
        
        # Проверка дубликатов
        if text in seen_texts:
            stats['duplicates'] += 1
            issues.append(f"  ⚠️ Запись #{idx}: Дубликат текста")
            continue
        
        seen_texts.add(text)
        
        # Проверка меток (io_type, emo_eval, fake_claim)
        labels_found = False
        
        # Проверяем разные форматы хранения меток
        if 'io_type' in item or 'emo_eval' in item or 'fake_claim' in item:
            labels_found = True
        elif 'annotations' in item and item['annotations']:
            # Формат Label Studio
            for annotation in item['annotations']:
                if 'result' in annotation and annotation['result']:
                    labels_found = True
                    break
        
        if not labels_found:
            stats['missing_labels'] += 1
            issues.append(f"  ⚠️ Запись #{idx}: Отсутствуют метки (io_type, emo_eval, fake_claim)")
        
        # Если есть проблемы, но текст валидный - пытаемся исправить
        if issues and text:
            # Нормализация структуры
            normalized = {
                'text': text.strip(),
                'source': item.get('source') or item.get('data', {}).get('source', 'Unknown'),
                'date': item.get('date') or item.get('data', {}).get('date', ''),
                'io_type': item.get('io_type'),
                'emo_eval': item.get('emo_eval'),
                'fake_claim': item.get('fake_claim')
            }
            
            # Попытка извлечь метки из annotations
            if 'annotations' in item and item['annotations']:
                for annotation in item['annotations']:
                    for result in annotation.get('result', []):
                        from_name = result.get('from_name')
                        value = result.get('value', {}).get('choices', [''])[0]
                        
                        if from_name == 'io_type' and not normalized['io_type']:
                            normalized['io_type'] = value
                        elif from_name == 'emo_eval' and not normalized['emo_eval']:
                            normalized['emo_eval'] = value
                        elif from_name == 'fake_claim' and not normalized['fake_claim']:
                            normalized['fake_claim'] = value
            
            cleaned_data.append(normalized)
            stats['fixed'].append(idx)
        
        elif not issues:
            # Запись корректная
            stats['valid'] += 1
            cleaned_data.append(item)
        
        # Вывод проблем
        for issue in issues:
            print(issue)
    
    # Итоговая статистика
    print("\n" + "="*50)
    print("📊 СТАТИСТИКА ПРОВЕРКИ")
    print("="*50)
    print(f"Всего записей:           {stats['total']}")
    print(f"✅ Корректных:           {stats['valid']}")
    print(f"🔧 Исправлено:           {len(stats['fixed'])}")
    print(f"⚠️  Пустой текст:         {stats['missing_text']}")
    print(f"⚠️  Дубликаты:            {stats['duplicates']}")
    print(f"⚠️  Без меток:            {stats['missing_labels']}")
    
    # Сохранение исправленной версии
    if cleaned_data and len(cleaned_data) < len(data):
        backup_file = Path('data/project.json.backup')
        
        print(f"\n💾 Создание резервной копии: {backup_file}")
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Сохранение исправленного файла...")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Файл исправлен! Удалено {len(data) - len(cleaned_data)} проблемных записей")
    
    elif len(cleaned_data) == len(data):
        print("\n✅ Все записи корректные! Исправления не требуются.")
    
    # Рекомендации
    print("\n" + "="*50)
    print("💡 РЕКОМЕНДАЦИИ")
    print("="*50)
    
    if stats['missing_text'] > 0:
        print("⚠️  Найдены записи с пустым текстом - они будут пропущены при импорте")
    
    if stats['duplicates'] > 0:
        print("⚠️  Найдены дубликаты - они были удалены из итогового файла")
    
    if stats['missing_labels'] > 0:
        print("⚠️  Некоторые записи без меток - модель не сможет обучиться на них")
        print("   Решение: Добавьте метки вручную или используйте только для тестирования")
    
    if cleaned_data:
        print(f"\n✅ Готово! Валидных записей для импорта: {len(cleaned_data)}")
        print("\n📌 СЛЕДУЮЩИЙ ШАГ:")
        print("   python migrate_to_sqlite.py")


if __name__ == "__main__":
    validate_and_fix_project_json()