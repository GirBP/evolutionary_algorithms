#!/usr/bin/env python3
"""scripts/publish_to_github.py — безпечна публікація репозиторію на GitHub.

Публікує поточний стан публічної копії дисертаційного репозиторію на
https://github.com/GirBP/evolutionary_algorithms.git.

Гарантії безпеки:
  - НІКОЛИ не видаляє .git (ні за яких умов, навіть при помилці).
  - Якщо git-репозиторій відсутній — виконує `git init` + перший коміт.
  - Якщо git-репозиторій вже існує — `git add -A` + новий коміт поверх
    наявної історії. Force-push можливий ЛИШЕ з явним прапорцем
    --replace-remote і окремим підтвердженням (--force-with-lease).
  - Перед будь-якою дією друкує повний план і питає підтвердження (input());
    push виконується лише після ОКРЕМОГО додаткового підтвердження.
  - Не містить токенів, паролів чи email — авторизація виконується
    стандартним git-механізмом користувача (SSH-ключ або credential helper).

Використання:
    python scripts/publish_to_github.py
    python scripts/publish_to_github.py --remote-url <url>   # інший remote
    python scripts/publish_to_github.py --branch main        # інша гілка
    python scripts/publish_to_github.py --message "..."      # свій текст коміту
    python scripts/publish_to_github.py --no-push            # лише init/commit, без push
    python scripts/publish_to_github.py --check-only         # лише перевірки, без git-дій
    python scripts/publish_to_github.py --replace-remote     # замінити вміст remote-гілки
                                                             # (якщо репозиторій на GitHub створено
                                                             #  із автозгенерованим README-коммітом)
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REMOTE_URL = "https://github.com/GirBP/evolutionary_algorithms.git"
DEFAULT_BRANCH = "main"
DEFAULT_MESSAGE = "Результати досліджень: код, графіки і експерименти"


def run(cmd: list[str], check: bool = False, capture: bool = False) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, check=check,
                           capture_output=capture)


def confirm(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes", "так", "т")


def check_git_available() -> None:
    if shutil.which("git") is None:
        print("ПОМИЛКА: git не знайдено в PATH. Встановіть git і повторіть спробу.")
        sys.exit(1)


def has_git_repo() -> bool:
    return (REPO_ROOT / ".git").exists()


def get_current_remote(name: str = "origin") -> str | None:
    result = run(["git", "remote", "get-url", name], capture=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def get_status_porcelain() -> str:
    result = run(["git", "status", "--porcelain"], capture=True)
    return result.stdout


TEXT_EXT = {".py", ".md", ".txt", ".sh", ".csv", ".tsv", ".json", ".jsonl", ".cfg", ".toml", ".yaml", ".yml", ".gitignore"}
SCAN_PATTERN = re.compile(r"\b(claude|anthropic|openai|chatgpt|gemini|copilot|antigravity)\b", re.I)


def scan_cleanliness() -> list[str]:
    """Перевіряє, що у файлах репозиторію немає згадок сторонніх інструментів розробки."""
    hits: list[str] = []
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue  # не сканувати власний файл (містить перелік патернів)
        if path.suffix.lower() not in TEXT_EXT and path.name != ".gitignore":
            continue
        try:
            if path.stat().st_size > 5_000_000:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if SCAN_PATTERN.search(line):
                hits.append(f"{path.relative_to(REPO_ROOT)}:{i}: {line.strip()[:120]}")
                if len(hits) >= 50:
                    return hits
    return hits


def resolve_git_identity(author_name: str | None, author_email: str | None) -> tuple[str, str] | None:
    """Визначає git-ідентичність для коміту: конфіг або прапорці --author-*."""
    name = run(["git", "config", "user.name"], capture=True).stdout.strip()
    email = run(["git", "config", "user.email"], capture=True).stdout.strip()
    if author_name:
        name = author_name
    if author_email:
        email = author_email
    if not name or not email:
        print("\n  ПОМИЛКА: git user.name / user.email не налаштовано.")
        print("  Або задайте глобально:")
        print("    git config --global user.name  \"Ваше Ім'я\"")
        print("    git config --global user.email \"ваша@пошта\"")
        print("  Або передайте прапорці цьому скрипту:")
        print("    --author-name \"Ваше Ім'я\" --author-email \"ваша@пошта\"")
        return None
    print(f"\n  Автор коміту: {name} <{email}>")
    return name, email


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--remote-url", default=DEFAULT_REMOTE_URL)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--message", default=DEFAULT_MESSAGE)
    parser.add_argument("--no-push", action="store_true", help="лише git init/commit, без push")
    parser.add_argument("--check-only", action="store_true", help="лише перевірки чистоти та плану, без git-дій")
    parser.add_argument("--onto-remote", action="store_true",
                        help="надбудувати коміт поверх наявної історії (зберігає перший коміт "
                             "і дату створення; звичайний push без force) — рекомендовано, якщо "
                             "репозиторій уже має коміт-заглушку")
    parser.add_argument("--replace-remote", action="store_true",
                        help="ЗАТЕРТИ історію віддаленої гілки вашим комітом (push --force-with-lease); "
                             "перший коміт буде втрачено — використовуйте лише якщо це прийнятно")
    parser.add_argument("--author-name", default=None, help="ім'я автора коміту (локальний git-конфіг репозиторію)")
    parser.add_argument("--author-email", default=None, help="email автора коміту (локальний git-конфіг репозиторію)")
    args = parser.parse_args()

    check_git_available()

    print("=" * 70)
    print("  ПУБЛІКАЦІЯ РЕПОЗИТОРІЮ НА GITHUB")
    print("=" * 70)
    print(f"  Корінь репозиторію: {REPO_ROOT}")
    print(f"  Remote URL:         {args.remote_url}")
    print(f"  Гілка:              {args.branch}")

    print("\n  Перевірка чистоти вмісту (згадки сторонніх інструментів розробки)...")
    hits = scan_cleanliness()
    if hits:
        print(f"  ЗНАЙДЕНО {len(hits)} згадок — публікацію зупинено:")
        for h in hits[:20]:
            print(f"    {h}")
        sys.exit(1)
    print("  Чисто: згадок сторонніх інструментів не знайдено.")

    identity = resolve_git_identity(args.author_name, args.author_email)
    if identity is None:
        sys.exit(1)

    if args.check_only:
        print("\n  --check-only: перевірки пройдено, git-дії не виконуються. Вихід.")
        return

    repo_exists = has_git_repo()
    plan: list[str] = []

    if not repo_exists:
        print("\n  .git відсутній — буде ініціалізовано новий локальний репозиторій.")
        plan.append("git init")
        plan.append(f"git checkout -b {args.branch}")
        plan.append(f"git remote add origin {args.remote_url}")
        plan.append("git add -A")
        plan.append(f'git commit -m "{args.message}"  (перший коміт)')
    else:
        print("\n  .git вже існує — НЕ буде видалено ні за яких умов; коміт додається поверх історії.")
        current_remote = get_current_remote("origin")
        if current_remote is None:
            plan.append(f"git remote add origin {args.remote_url}")
        elif current_remote != args.remote_url:
            print(f"\n  УВАГА: origin вже вказує на інший URL:")
            print(f"    поточний:    {current_remote}")
            print(f"    очікуваний:  {args.remote_url}")
            if not confirm("  Продовжити з наявним origin (без зміни remote)?"):
                print("Скасовано користувачем.")
                sys.exit(0)

        status = get_status_porcelain()
        changed_lines = [ln for ln in status.splitlines() if ln.strip()]
        if not changed_lines:
            print("\n  Робоча копія чиста — немає змін для коміту.")
        else:
            print(f"\n  Знайдено {len(changed_lines)} змінених/нових файлів (git status --porcelain):")
            for line in changed_lines[:40]:
                print(f"    {line}")
            if len(changed_lines) > 40:
                print(f"    ... і ще {len(changed_lines) - 40}")
            plan.append("git add -A")
            plan.append(f'git commit -m "{args.message}"')

    if not args.no_push:
        plan.append(f"git push -u origin {args.branch}")

    if not plan:
        print("\n  Немає дій для виконання (репозиторій вже синхронізовано). Вихід.")
        return

    print("\n  ПЛАН ДІЙ:")
    for i, step in enumerate(plan, 1):
        print(f"    {i}. {step}")

    print("\n  Аутентифікація для push виконується git-механізмом користувача")
    print("  (SSH-ключ або збережений credential helper) — цей скрипт не запитує")
    print("  і не зберігає жодних токенів, паролів чи персональних даних.")

    if not confirm("\nВиконати план дій?"):
        print("Скасовано користувачем. Жодних змін не внесено.")
        sys.exit(0)

    try:
        if not repo_exists:
            run(["git", "init"], check=True)
            if args.author_name:
                run(["git", "config", "user.name", args.author_name], check=True)
            if args.author_email:
                run(["git", "config", "user.email", args.author_email], check=True)
            run(["git", "checkout", "-b", args.branch])
            run(["git", "remote", "add", "origin", args.remote_url], check=True)
            run(["git", "add", "-A"], check=True)
            commit_result = run(["git", "commit", "-m", args.message])
            if commit_result.returncode != 0:
                print("  Коміт не створено (можливо, немає файлів). Перевірте вивід вище.")
        else:
            current_remote = get_current_remote("origin")
            if current_remote is None:
                run(["git", "remote", "add", "origin", args.remote_url], check=True)
            status = get_status_porcelain()
            if status.strip():
                run(["git", "add", "-A"], check=True)
                commit_result = run(["git", "commit", "-m", args.message])
                if commit_result.returncode != 0:
                    print("  Коміт не створено. Перевірте вивід вище.")
    except subprocess.CalledProcessError as exc:
        print(f"\n  ПОМИЛКА під час виконання '{' '.join(exc.cmd)}' (код {exc.returncode}).")
        print("  .git та локальні файли НЕ видалено. Виправте проблему і запустіть скрипт знову.")
        sys.exit(1)

    if args.no_push:
        print("\n  Готово (push пропущено за прапорцем --no-push).")
        return

    if args.onto_remote:
        print("\n  РЕЖИМ --onto-remote: ваш коміт буде НАДБУДОВАНО поверх наявної історії.")
        print("  Перший коміт репозиторію та дата створення зберігаються без змін.")
        remote_info = run(["git", "ls-remote", args.remote_url, args.branch], capture=True)
        if not remote_info.stdout.strip():
            print("  На віддаленій гілці ще нема комітів — зберігати нічого; використайте звичайний режим.")
            sys.exit(1)
        print(f"  Наявна remote-гілка (лишиться в історії): {remote_info.stdout.strip()[:60]}")
        if not confirm(f"\nНадбудувати ваш коміт поверх історії {args.remote_url} (гілка {args.branch})?"):
            print("Push скасовано користувачем. Локальні коміти збережено, нічого не відправлено.")
            return
        print("\n  Отримання наявної історії (git fetch)...")
        run(["git", "fetch", "origin", args.branch], check=True)
        # HEAD → на квітневий коміт, але наш повний дерево-знімок лишається в індексі:
        run(["git", "reset", "--soft", f"origin/{args.branch}"], check=True)
        # новий коміт поверх наявного першого коміту (звичайний батьківський зв'язок):
        commit_result = run(["git", "commit", "-m", args.message])
        if commit_result.returncode != 0:
            print("  Немає змін відносно наявної історії — нічого коммітити.")
            return
        # звичайний fast-forward push, force НЕ потрібен:
        push_result = run(["git", "push", "-u", "origin", args.branch])
    elif args.replace_remote:
        print("\n  РЕЖИМ --replace-remote: вміст віддаленої гілки буде ЗАМІНЕНО вашим комітом.")
        print("  УВАГА: цей режим ЗАТИРАЄ перший коміт (для збереження історії — режим --onto-remote).")
        remote_info = run(["git", "ls-remote", args.remote_url, args.branch], capture=True)
        if remote_info.stdout.strip():
            print(f"  Стан remote-гілки, який буде замінено: {remote_info.stdout.strip()[:60]}")
        if not confirm(f"\nЗамінити вміст {args.remote_url} (гілка {args.branch}) вашим комітом?"):
            print("Push скасовано користувачем. Локальні коміти збережено, нічого не відправлено.")
            return
        # Для коректного --force-with-lease потрібна remote-tracking-гілка:
        # у щойно ініціалізованому репозиторії її нема, тому спершу fetch.
        print("\n  Отримання поточного стану remote (git fetch) для безпечної заміни...")
        run(["git", "fetch", "origin", args.branch])
        push_result = run(["git", "push", "--force-with-lease", "-u", "origin", args.branch])
        if push_result.returncode != 0:
            print("\n  Безпечна заміна (--force-with-lease) не вдалася. Причина зазвичай — "
                  "розбіжність історій щойно створеного репозиторію. Спробувати пряму "
                  "заміну (git push --force)?")
            if confirm("  Виконати git push --force?"):
                push_result = run(["git", "push", "--force", "-u", "origin", args.branch])
    else:
        if not confirm(f"\nВиконати push у {args.remote_url} (гілка {args.branch})?"):
            print("Push скасовано користувачем. Локальні коміти збережено, нічого не відправлено.")
            return
        push_result = run(["git", "push", "-u", "origin", args.branch])
    if push_result.returncode != 0:
        print("\n  ПОМИЛКА push. Перевірте автентифікацію (SSH-ключ / credential helper) "
              "та права доступу до репозиторію на GitHub. Локальні коміти НЕ втрачено.")
        sys.exit(1)

    print("\n  Готово: зміни опубліковано.")


if __name__ == "__main__":
    main()
