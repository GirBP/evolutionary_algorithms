# common/dependencies.py — перевірка та автоматичне встановлення залежностей

import subprocess
import sys
from typing import Dict, List, Tuple, Optional


def check_and_install_package(package_name: str, import_name: Optional[str] = None,
                             install_name: Optional[str] = None) -> bool:
    """
    Перевіряє наявність пакету та автоматично встановлює його, якщо відсутній.

    Args:
        package_name: Назва пакету для перевірки (наприклад, "adabelief_pytorch")
        import_name: Назва для імпорту (якщо відрізняється від package_name)
        install_name: Назва для pip install (якщо відрізняється від package_name)

    Returns:
        True якщо пакет доступний (був встановлений або вже існував), False якщо не вдалося встановити
    """
    if import_name is None:
        import_name = package_name
    if install_name is None:
        install_name = package_name.replace('_', '-')

    # Перевіряємо, чи можна імпортувати
    try:
        __import__(import_name)
        return True
    except ImportError:
        pass

    # Спробуємо встановити
    print(f"[DEPENDENCIES] Пакет '{import_name}' не знайдено. Спробуємо встановити '{install_name}'...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", install_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        print(f"[DEPENDENCIES]  Пакет '{install_name}' успішно встановлено")

        # Перевіряємо після встановлення
        try:
            __import__(import_name)
            return True
        except ImportError:
            print(f"[DEPENDENCIES]   Пакет '{install_name}' встановлено, але імпорт '{import_name}' все ще не працює")
            return False

    except subprocess.CalledProcessError as e:
        print(f"[DEPENDENCIES]  Помилка встановлення '{install_name}': {e}")
        print(f"[DEPENDENCIES] Спробуйте встановити вручну: pip install {install_name}")
        return False
    except Exception as e:
        print(f"[DEPENDENCIES]  Несподівана помилка при встановленні '{install_name}': {e}")
        return False


def check_dependencies(dependencies: List[Tuple[str, Optional[str], Optional[str]]]) -> Dict[str, bool]:
    """
    Перевіряє та встановлює список залежностей.

    Args:
        dependencies: Список кортежів (package_name, import_name, install_name)
                     де import_name та install_name можуть бути None

    Returns:
        Словник {package_name: is_available} з результатами перевірки
    """
    results = {}

    for dep in dependencies:
        if len(dep) == 1:
            package_name, import_name, install_name = dep[0], None, None
        elif len(dep) == 2:
            package_name, import_name = dep[0], dep[1]
            install_name = None
        else:
            package_name, import_name, install_name = dep[0], dep[1], dep[2]

        results[package_name] = check_and_install_package(package_name, import_name, install_name)

    return results


def ensure_basic_dependencies():
    """
    Перевіряє та встановлює базові залежності, необхідні для всіх експериментів.
    Ці залежності потрібні для базової функціональності.

    Returns:
        Словник з результатами перевірки для кожного пакету
    """
    dependencies = [
        # (package_name, import_name, install_name)
        ("tqdm", "tqdm", "tqdm"),
        ("cma", "cma", "cma"),
        ("scipy", "scipy", "scipy"),
    ]

    return check_dependencies(dependencies)


def ensure_experiment_dependencies():
    """
    Перевіряє та встановлює залежності, необхідні для всіх експериментів.
    Спочатку перевіряє базові залежності, потім опціональні.

    Returns:
        Словник з результатами перевірки для кожного пакету
    """
    # Спочатку перевіряємо базові залежності
    basic_results = ensure_basic_dependencies()

    # Потім опціональні залежності для різних експериментів
    optional_dependencies = [
        # (package_name, import_name, install_name)
        ("adabelief_pytorch", "adabelief_pytorch", "adabelief-pytorch"),
        ("lion_pytorch", "lion_pytorch", "lion-pytorch"),
        # Залежності для Ex06-Ex13
        ("torchvision", "torchvision", "torchvision"),
        ("optuna", "optuna", "optuna"),
    ]

    optional_results = check_dependencies(optional_dependencies)

    # Об'єднуємо результати
    return {**basic_results, **optional_results}


def safe_import(module_name: str, package_name: Optional[str] = None,
                install_name: Optional[str] = None, default=None):
    """
    Безпечний імпорт модуля з автоматичним встановленням, якщо відсутній.

    Args:
        module_name: Назва модуля для імпорту
        package_name: Назва пакету для перевірки (за замовчуванням = module_name)
        install_name: Назва для pip install (за замовчуванням = module_name.replace('_', '-'))
        default: Значення за замовчуванням, якщо модуль недоступний

    Returns:
        Імпортований модуль або default значення
    """
    if package_name is None:
        package_name = module_name
    if install_name is None:
        install_name = module_name.replace('_', '-')

    is_available = check_and_install_package(package_name, module_name, install_name)

    if is_available:
        try:
            return __import__(module_name)
        except ImportError:
            return default
    else:
        return default
