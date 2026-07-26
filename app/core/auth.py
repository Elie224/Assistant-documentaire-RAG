from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import time
import uuid
from pathlib import Path


class AuthError(ValueError):
    pass


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${salt.hex()}${digest.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, salt_hex, digest_hex = encoded.split("$", 2)
        if algorithm != "scrypt":
            return False
        actual = _hash_password(password, bytes.fromhex(salt_hex)).split("$", 2)[2]
        return hmac.compare_digest(actual, digest_hex)
    except (ValueError, TypeError):
        return False


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        """CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            workspace_id TEXT UNIQUE NOT NULL,
            created_at REAL NOT NULL
        )"""
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(user_id),
            expires_at REAL NOT NULL
        )"""
    )
    connection.commit()
    return connection


def register(path: Path, email: str, password: str) -> tuple[str, str]:
    normalized = email.strip().casefold()
    if "@" not in normalized:
        raise AuthError("Adresse email invalide.")
    user_id = uuid.uuid4().hex
    workspace_id = f"user-{user_id[:16]}"
    with _connect(path) as connection:
        try:
            connection.execute(
                "INSERT INTO users VALUES (?, ?, ?, ?, ?)",
                (user_id, normalized, _hash_password(password), workspace_id, time.time()),
            )
            connection.commit()
        except sqlite3.IntegrityError as error:
            raise AuthError("Cette adresse email est déjà utilisée.") from error
    return user_id, workspace_id


def login(path: Path, email: str, password: str, ttl_hours: int) -> tuple[str, str, str, int]:
    normalized = email.strip().casefold()
    with _connect(path) as connection:
        row = connection.execute(
            "SELECT user_id, password_hash, workspace_id FROM users WHERE email = ?",
            (normalized,),
        ).fetchone()
        if not row or not _verify_password(password, row[1]):
            raise AuthError("Identifiants invalides.")
        token = secrets.token_urlsafe(32)
        expires_at = int(time.time()) + ttl_hours * 3600
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?)",
            (hashlib.sha256(token.encode()).hexdigest(), row[0], expires_at),
        )
        connection.commit()
    return token, row[0], row[2], ttl_hours * 3600


def authenticate(path: Path, token: str) -> tuple[str, str, str] | None:
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    with _connect(path) as connection:
        row = connection.execute(
            """SELECT users.user_id, users.email, users.workspace_id, sessions.expires_at
               FROM sessions JOIN users ON users.user_id = sessions.user_id
               WHERE sessions.token_hash = ?""",
            (token_hash,),
        ).fetchone()
        if not row:
            return None
        if row[3] <= time.time():
            connection.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
            connection.commit()
            return None
    return row[0], row[1], row[2]



def revoke(path: Path, token: str) -> None:
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    with _connect(path) as connection:
        connection.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
        connection.commit()



def change_password(path: Path, user_id: str, current_password: str, new_password: str) -> None:
    with _connect(path) as connection:
        row = connection.execute(
            "SELECT password_hash FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        if not row or not _verify_password(current_password, row[0]):
            raise AuthError("Mot de passe actuel invalide.")
        connection.execute(
            "UPDATE users SET password_hash = ? WHERE user_id = ?",
            (_hash_password(new_password), user_id),
        )
        connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        connection.commit()
