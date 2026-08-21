# Модуль перевірки та встановлення залежностей

Модуль `common/dependencies.py` надає функції для автоматичної перевірки та встановлення відсутніх Python пакетів.

## Основні функції

### `check_and_install_package(package_name, import_name=None, install_name=None)`

Перевіряє наявність пакету та автоматично встановлює його через pip, якщо відсутній.

**Параметри:**
- `package_name`: Назва пакету для перевірки
- `import_name`: Назва для імпорту (якщо відрізняється від package_name)
- `install_name`: Назва для pip install (якщо відрізняється від package_name)

**Повертає:** `True` якщо пакет доступний, `False` якщо не вдалося встановити

**Приклад:**
```python
from common import check_and_install_package

# Простий випадок (import_name та install_name виводяться автоматично)
is_available = check_and_install_package("adabelief_pytorch")

# З явним вказанням назв
is_available = check_and_install_package(
    package_name="adabelief_pytorch",
    import_name="adabelief_pytorch",
    install_name="adabelief-pytorch"
)
```

### `check_dependencies(dependencies)`

Перевіряє та встановлює список залежностей.

**Параметри:**
- `dependencies`: Список кортежів `(package_name, import_name, install_name)` або `(package_name, import_name)` або `(package_name,)`

**Повертає:** Словник `{package_name: is_available}` з результатами перевірки

**Приклад:**
```python
from common import check_dependencies

deps = [
    ("adabelief_pytorch", "adabelief_pytorch", "adabelief-pytorch"),
    ("lion_pytorch", "lion_pytorch", "lion-pytorch"),
]
results = check_dependencies(deps)
# results = {"adabelief_pytorch": True, "lion_pytorch": False}
```

### `ensure_experiment_dependencies()`

Перевіряє та встановлює залежності, необхідні для експериментів Ex01 та Ex02.

**Повертає:** Словник з результатами перевірки для кожного пакету

**Приклад:**
```python
from common import ensure_experiment_dependencies

deps_status = ensure_experiment_dependencies()
# deps_status = {"adabelief_pytorch": True, "lion_pytorch": True}
```

### `safe_import(module_name, package_name=None, install_name=None, default=None)`

Безпечний імпорт модуля з автоматичним встановленням, якщо відсутній.

**Параметри:**
- `module_name`: Назва модуля для імпорту
- `package_name`: Назва пакету для перевірки
- `install_name`: Назва для pip install
- `default`: Значення за замовчуванням, якщо модуль недоступний

**Повертає:** Імпортований модуль або `default` значення

**Приклад:**
```python
from common import safe_import

AdaBelief = safe_import(
    "adabelief_pytorch",
    package_name="adabelief_pytorch",
    install_name="adabelief-pytorch",
    default=None
)

if AdaBelief is not None:
    from adabelief_pytorch import AdaBelief
```

## Використання в експериментах

> Примітка: приклади нижче згадують також Ex02; до публічної копії включено лише Ex01 і Ex03 (папка `Ex01-03_CMA_Boundary/`), Ex02 не включено — приклад для нього суто ілюстративний.

### Ex01 та Ex02

Обидва експерименти використовують `ensure_experiment_dependencies()` для автоматичної перевірки та встановлення:
- `adabelief-pytorch`
- `lion-pytorch`

**Приклад використання в ex02.py:**
```python
from common import ensure_experiment_dependencies

# Перевіряємо та встановлюємо залежності
deps_status = ensure_experiment_dependencies()

ADABELIEF_AVAILABLE = deps_status.get("adabelief_pytorch", False)
LION_AVAILABLE = deps_status.get("lion_pytorch", False)

if ADABELIEF_AVAILABLE:
    from adabelief_pytorch import AdaBelief
else:
    AdaBelief = None
```

## Поведінка

1. **Якщо пакет вже встановлений**: Функція просто перевіряє імпорт та повертає `True`
2. **Якщо пакет відсутній**: Функція спробує встановити його через `pip install -q <package_name>`
3. **Після встановлення**: Функція перевіряє імпорт ще раз та повертає результат
4. **У разі помилки**: Функція виводить повідомлення про помилку та повертає `False`

## Переваги

-  Автоматичне встановлення відсутніх залежностей
-  Тихий режим (`-q` flag для pip)
-  Інформативні повідомлення про статус встановлення
-  Безпечна обробка помилок
-  Можливість використання fallback значень

## Примітки

- Функції використовують `subprocess.check_call()` для виклику pip
- Встановлення виконується в тихому режимі (`-q` flag)
- Помилки встановлення логуються, але не переривають виконання програми
- Рекомендується використовувати в початку скриптів перед основним кодом
