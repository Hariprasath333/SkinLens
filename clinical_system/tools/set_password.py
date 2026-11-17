"""Create a staff account or set a new password (you type it; it is never shown or saved in plain text).
If the username does not exist yet, it is created. The first account ever created is the Admin.

Run from the clinical_system folder:
    .\\.venv\\Scripts\\python.exe tools\\set_password.py            # list accounts
    .\\.venv\\Scripts\\python.exe tools\\set_password.py <username> # set a new password
"""
from __future__ import annotations

import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webapp.db import SessionLocal, User, init_db  # noqa: E402
from webapp.main import hash_pw  # noqa: E402


def strong(pw: str) -> str | None:
    if len(pw) < 12:
        return "at least 12 characters"
    kinds = sum([any(c.islower() for c in pw), any(c.isupper() for c in pw),
                 any(c.isdigit() for c in pw), any(not c.isalnum() for c in pw)])
    if kinds < 3:
        return "use at least 3 of: lowercase, UPPERCASE, digits, symbols"
    return None


def main() -> None:
    init_db()
    db = SessionLocal()
    try:
        if len(sys.argv) < 2:
            print("Accounts:")
            for u in db.query(User).order_by(User.id):
                print(f"  {u.username:<24} {u.role:<7} {u.full_name}")
            print("\nRun again with a username to change its password.")
            return
        username = sys.argv[1].strip().lower()
        u = db.query(User).filter_by(username=username).one_or_none()
        if u is None:
            if not (3 <= len(username) <= 64) or not username.replace(".", "").replace("_", "").isalnum():
                sys.exit("Username: 3-64 characters, letters, numbers, dot or underscore.")
            first = db.query(User).count() == 0
            role = "admin" if first else "doctor"
            print(f"No account called {username!r} yet, so a new {role.upper()} account will be created.")
            full_name = ""
            while not full_name:
                full_name = input("Full name shown in the app (e.g. Dr. Hari Prasath): ").strip()[:120]
            u = User(username=username, full_name=full_name, role=role, password_hash="!")
            db.add(u)
        while True:
            pw = getpass.getpass(f"New password for {u.username}: ")
            problem = strong(pw)
            if problem:
                print("Too weak:", problem)
                continue
            if getpass.getpass("Type it again: ") != pw:
                print("They did not match, try again.")
                continue
            break
        u.password_hash = hash_pw(pw)
        db.commit()
        print(f"Password saved for {u.username} ({u.role}).")
    finally:
        db.close()


if __name__ == "__main__":
    main()
