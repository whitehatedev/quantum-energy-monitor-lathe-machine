"""
firebase_client.py
Firebase Email/Password authentication + Realtime Database REST helpers.
"""

import time
import requests
from typing import Any, Optional

# ============================================================
#  CONFIG — from your ESP32 Arduino code
# ============================================================
FIREBASE_WEB_API_KEY = "AIzaSyC2A3qi7vfPHr83Kev2TGyJBiSGMM7kmYs"
DATABASE_URL = "https://new-project-210e3-default-rtdb.firebaseio.com"
USER_EMAIL = "sb284160@gmail.com"
USER_PASSWORD = "Password@1"


class FirebaseClient:
    """Firebase Auth + RTDB client using REST API."""

    def __init__(self, api_key: str = FIREBASE_WEB_API_KEY,
                 database_url: str = DATABASE_URL,
                 email: str = USER_EMAIL,
                 password: str = USER_PASSWORD):
        self.api_key = api_key
        self.database_url = database_url.rstrip("/")
        self.email = email
        self.password = password
        self.id_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expiry: float = 0.0
        self.user_uid: Optional[str] = None
        self._sign_in()

    # ------------------------------------------------------------
    #  Authentication
    # ------------------------------------------------------------
    def _sign_in(self) -> None:
        url = (
            "https://identitytoolkit.googleapis.com/v1/accounts:"
            f"signInWithPassword?key={self.api_key}"
        )
        payload = {
            "email": self.email,
            "password": self.password,
            "returnSecureToken": True,
        }
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Firebase sign-in failed: {resp.status_code} — {resp.text}"
            )
        data = resp.json()
        self.id_token = data["idToken"]
        self.refresh_token = data["refreshToken"]
        self.user_uid = data["localId"]
        self.token_expiry = time.time() + int(data.get("expiresIn", 3600)) - 60
        print(f"✅ Signed in as {self.email} (uid={self.user_uid})")

    def _refresh_id_token(self) -> None:
        url = f"https://securetoken.googleapis.com/v1/token?key={self.api_key}"
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
        }
        resp = requests.post(url, data=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        self.id_token = data["id_token"]
        self.refresh_token = data["refresh_token"]
        self.token_expiry = time.time() + int(data.get("expires_in", 3600)) - 60
        print("🔄 Firebase ID token refreshed.")

    def _ensure_valid_token(self) -> None:
        if not self.id_token or time.time() >= self.token_expiry:
            self._refresh_id_token()

    # ------------------------------------------------------------
    #  Realtime Database operations
    # ------------------------------------------------------------
    def get(self, path: str) -> Any:
        self._ensure_valid_token()
        url = f"{self.database_url}/{path.lstrip('/')}.json"
        params = {"auth": self.id_token}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def put(self, path: str, data: Any) -> Any:
        self._ensure_valid_token()
        url = f"{self.database_url}/{path.lstrip('/')}.json"
        params = {"auth": self.id_token}
        resp = requests.put(url, params=params, json=data, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def update(self, path: str, data: dict) -> Any:
        self._ensure_valid_token()
        url = f"{self.database_url}/{path.lstrip('/')}.json"
        params = {"auth": self.id_token}
        resp = requests.patch(url, params=params, json=data, timeout=15)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------
    #  Convenience reader for this project
    # ------------------------------------------------------------
    def fetch_all_sensors(self) -> dict:
        return self.get("/") or {}


if __name__ == "__main__":
    fb = FirebaseClient()
    data = fb.fetch_all_sensors()
    print("\n📡 Current Firebase snapshot:")
    import json
    print(json.dumps(data, indent=2))