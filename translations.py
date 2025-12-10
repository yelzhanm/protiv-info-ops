# translations.py
"""
Переводы интерфейса на казахский, русский и английский
"""

TRANSLATIONS = {
    'kk': {
        # Общие
        'app_title': 'Ақпараттық операцияларға қарсы жүйе',
        'welcome': 'Қош келдіңіз',
        'logout': 'Шығу',
        
        # Роли
        'admin': 'Администратор',
        'analyst': 'Аналитик',
        'linguist': 'Лингвист',
        'choose_role': 'Рөліңізді таңдаңыз',
        
        # Кнопки
        'search': 'Іздеу',
        'add': 'Қосу',
        'delete': 'Өшіру',
        'edit': 'Өзгерту',
        'save': 'Сақтау',
        'cancel': 'Болдырмау',
        'analyze': 'Талдау',
        'export': 'Экспорт',
        'import': 'Импорт',
        'filter': 'Сүзгі',
        
        # Формы
        'text': 'Мәтін',
        'source': 'Дереккөз',
        'channel': 'Канал',
        'date': 'Күні',
        'language': 'Тіл',
        'term': 'Термин',
        'scope_note': 'Анықтама',
        
        # Админ панель
        'admin_panel': 'Администратор панелі',
        'message_feed': 'Хабарламалар ағыны',
        'nlp_analysis': 'NLP талдауы',
        'message_analysis': 'Хабарлама талдауы',
        'enter_text': 'Талдауға арналған мәтінді енгізіңіз',
        'analyzing': 'Талдау жүргізілуде...',
        'analysis_complete': 'Талдау аяқталды',
        
        # Аналитика
        'analytics': 'Аналитика',
        'overview': 'Жалпы шолу',
        'total_records': 'Жалпы жазбалар',
        'filtered_records': 'Сүзілген жазбалар',
        'date_range': 'Күндер аралығы',
        'avg_length': 'Орташа ұзындық',
        'time_series': 'Уақыт бойынша тренд',
        'top_sources': 'Топ дереккөздер',
        'sentiment_analysis': 'Тон талдауы',
        'word_cloud': 'Сөздер бұлты',
        
        # Тезаурус
        'thesaurus': 'Тезаурус',
        'military_thesaurus': 'Әскери терминдер тезауруы',
        'search_term': 'Терминді іздеу',
        'add_term': 'Термин қосу',
        'term_name': 'Термин атауы',
        'broader_term': 'Кеңірек термин',
        'narrower_term': 'Тарырақ термин',
        'related_term': 'Байланысты термин',
        'language_equivalent': 'Тілдік эквивалент',
        
        # Анализ результаты
        'io_type': 'АО түрі',
        'fake_claim': 'Фейк деректеме',
        'emo_eval': 'Эмоционалдық бағалау',
        'named_entities': 'Атаулар',
        'military_terms': 'Әскери терминдер',
        'llm_summary': 'LLM қорытындысы',
        'threat_level': 'Қауіп деңгейі',
        
        # Уведомления
        'success': 'Сәтті орындалды',
        'error': 'Қате',
        'confirm_delete': 'Өшіруді растайсыз ба?',
        'yes': 'Иә',
        'no': 'Жоқ',
        
        # Статусы
        'positive': 'Позитивті',
        'negative': 'Негативті',
        'neutral': 'Нейтралды',
        'true': 'Рас',
        'false': 'Жалған',
    },
    
    'ru': {
        # Общие
        'app_title': 'Система противодействия информационным операциям',
        'welcome': 'Добро пожаловать',
        'logout': 'Выход',
        
        # Роли
        'admin': 'Администратор',
        'analyst': 'Аналитик',
        'linguist': 'Лингвист',
        'choose_role': 'Выберите роль',
        
        # Кнопки
        'search': 'Поиск',
        'add': 'Добавить',
        'delete': 'Удалить',
        'edit': 'Изменить',
        'save': 'Сохранить',
        'cancel': 'Отмена',
        'analyze': 'Анализ',
        'export': 'Экспорт',
        'import': 'Импорт',
        'filter': 'Фильтр',
        
        # Формы
        'text': 'Текст',
        'source': 'Источник',
        'channel': 'Канал',
        'date': 'Дата',
        'language': 'Язык',
        'term': 'Термин',
        'scope_note': 'Примечание',
        
        # Админ панель
        'admin_panel': 'Панель администратора',
        'message_feed': 'Лента сообщений',
        'nlp_analysis': 'NLP-анализ',
        'message_analysis': 'Анализ сообщения',
        'enter_text': 'Введите текст для анализа',
        'analyzing': 'Анализ выполняется...',
        'analysis_complete': 'Анализ завершен',
        
        # Аналитика
        'analytics': 'Аналитика',
        'overview': 'Обзор',
        'total_records': 'Всего записей',
        'filtered_records': 'Отфильтровано',
        'date_range': 'Период',
        'avg_length': 'Средняя длина',
        'time_series': 'Динамика',
        'top_sources': 'Топ источников',
        'sentiment_analysis': 'Анализ тональности',
        'word_cloud': 'Облако слов',
        
        # Тезаурус
        'thesaurus': 'Тезаурус',
        'military_thesaurus': 'Военный тезаурус',
        'search_term': 'Поиск термина',
        'add_term': 'Добавить термин',
        'term_name': 'Название термина',
        'broader_term': 'Более широкий термин',
        'narrower_term': 'Более узкий термин',
        'related_term': 'Связанный термин',
        'language_equivalent': 'Языковой эквивалент',
        
        # Результаты анализа
        'io_type': 'Тип ИО',
        'fake_claim': 'Фейк',
        'emo_eval': 'Эмоциональная оценка',
        'named_entities': 'Именованные сущности',
        'military_terms': 'Военные термины',
        'llm_summary': 'Резюме LLM',
        'threat_level': 'Уровень угрозы',
        
        # Уведомления
        'success': 'Успешно выполнено',
        'error': 'Ошибка',
        'confirm_delete': 'Подтвердите удаление',
        'yes': 'Да',
        'no': 'Нет',
        
        # Статусы
        'positive': 'Позитивный',
        'negative': 'Негативный',
        'neutral': 'Нейтральный',
        'true': 'Правда',
        'false': 'Ложь',
    },
    
    'en': {
        # General
        'app_title': 'Information Operations Countermeasure System',
        'welcome': 'Welcome',
        'logout': 'Logout',
        
        # Roles
        'admin': 'Administrator',
        'analyst': 'Analyst',
        'linguist': 'Linguist',
        'choose_role': 'Choose your role',
        
        # Buttons
        'search': 'Search',
        'add': 'Add',
        'delete': 'Delete',
        'edit': 'Edit',
        'save': 'Save',
        'cancel': 'Cancel',
        'analyze': 'Analyze',
        'export': 'Export',
        'import': 'Import',
        'filter': 'Filter',
        
        # Forms
        'text': 'Text',
        'source': 'Source',
        'channel': 'Channel',
        'date': 'Date',
        'language': 'Language',
        'term': 'Term',
        'scope_note': 'Scope Note',
        
        # Admin Panel
        'admin_panel': 'Admin Panel',
        'message_feed': 'Message Feed',
        'nlp_analysis': 'NLP Analysis',
        'message_analysis': 'Message Analysis',
        'enter_text': 'Enter text to analyze',
        'analyzing': 'Analyzing...',
        'analysis_complete': 'Analysis Complete',
        
        # Analytics
        'analytics': 'Analytics',
        'overview': 'Overview',
        'total_records': 'Total Records',
        'filtered_records': 'Filtered',
        'date_range': 'Date Range',
        'avg_length': 'Average Length',
        'time_series': 'Time Series',
        'top_sources': 'Top Sources',
        'sentiment_analysis': 'Sentiment Analysis',
        'word_cloud': 'Word Cloud',
        
        # Thesaurus
        'thesaurus': 'Thesaurus',
        'military_thesaurus': 'Military Thesaurus',
        'search_term': 'Search Term',
        'add_term': 'Add Term',
        'term_name': 'Term Name',
        'broader_term': 'Broader Term',
        'narrower_term': 'Narrower Term',
        'related_term': 'Related Term',
        'language_equivalent': 'Language Equivalent',
        
        # Analysis Results
        'io_type': 'IO Type',
        'fake_claim': 'Fake Claim',
        'emo_eval': 'Emotional Evaluation',
        'named_entities': 'Named Entities',
        'military_terms': 'Military Terms',
        'llm_summary': 'LLM Summary',
        'threat_level': 'Threat Level',
        
        # Notifications
        'success': 'Success',
        'error': 'Error',
        'confirm_delete': 'Confirm deletion',
        'yes': 'Yes',
        'no': 'No',
        
        # Statuses
        'positive': 'Positive',
        'negative': 'Negative',
        'neutral': 'Neutral',
        'true': 'True',
        'false': 'False',
    }
}

def get_translation(lang='kk'):
    """Получить словарь переводов для языка"""
    return TRANSLATIONS.get(lang, TRANSLATIONS['kk'])