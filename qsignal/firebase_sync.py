"""
Firebase Cloud Sync for Q-Signal.
Enables real-time syncing of traffic states, queue statistics, and emergency
vehicle dispatches to Firebase Firestore using the project service account.
"""

from __future__ import annotations

import os
import json
from typing import Dict, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_firebase_initialized = False
_db = None
_last_error: Optional[str] = None


def get_last_error() -> Optional[str]:
    """Returns the most recent Firebase error message, if any."""
    return _last_error


def get_firebase_credentials_path() -> Optional[str]:
    """Resolves path to service account credentials JSON."""
    env_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
    if env_path and os.path.isabs(env_path) and os.path.exists(env_path):
        return env_path
    
    # Check relative to project root
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if env_path:
        candidate = os.path.join(root, env_path)
        if os.path.exists(candidate):
            return candidate

    # Look for any firebase service account file in root
    for f in os.listdir(root):
        if f.endswith(".json") and "firebase-adminsdk" in f:
            return os.path.join(root, f)
            
    return None


def init_firebase() -> Dict[str, Any]:
    """
    Initializes the Firebase Admin SDK if credentials are valid.
    Returns status dict {success: bool, project_id: str, error: str}.
    """
    global _firebase_initialized, _db, _last_error
    if _firebase_initialized:
        return {"success": True, "project_id": os.getenv("FIREBASE_PROJECT_ID", "traffic-9aac8"), "error": None}

    cred_path = get_firebase_credentials_path()
    if not cred_path or not os.path.exists(cred_path):
        _last_error = "Service account JSON not found"
        return {"success": False, "project_id": None, "error": _last_error}

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        
        _db = firestore.client()
        _firebase_initialized = True
        _last_error = None
        
        # Read project id from JSON
        with open(cred_path, "r") as f:
            data = json.load(f)
            project_id = data.get("project_id", "traffic-9aac8")

        return {"success": True, "project_id": project_id, "error": None}
    except Exception as exc:
        _last_error = str(exc)
        return {"success": False, "project_id": None, "error": str(exc)}


def test_firestore_connection() -> tuple[bool, str]:
    """
    Tests live read/write access to Firestore.
    Returns (success: bool, message: str).
    """
    global _db, _last_error
    if not _firebase_initialized or _db is None:
        init_res = init_firebase()
        if not init_res["success"]:
            msg = f"Firebase initialization failed: {init_res.get('error')}"
            _last_error = msg
            return False, msg

    try:
        import time
        doc_ref = _db.collection("traffic_live").document("connection_test")
        doc_ref.set({"status": "connected", "tested_at": time.time()})
        _last_error = None
        return True, "Successfully connected and verified Firestore read/write access!"
    except Exception as exc:
        msg = str(exc)
        _last_error = msg
        return False, msg


def sync_traffic_snapshot(state_dict: Dict[str, Any]) -> bool:
    """Syncs a snapshot of traffic state to Firestore under 'traffic_live'."""
    global _db, _last_error
    if not _firebase_initialized or _db is None:
        return False

    try:
        doc_ref = _db.collection("traffic_live").document("latest")
        doc_ref.set(state_dict, merge=True)
        _last_error = None
        return True
    except Exception as exc:
        _last_error = str(exc)
        return False


def log_emergency_dispatch(event_dict: Dict[str, Any]) -> bool:
    """Logs an emergency ambulance corridor event to Firestore."""
    global _db, _last_error
    if not _firebase_initialized or _db is None:
        return False

    try:
        _db.collection("emergency_dispatches").add(event_dict)
        _last_error = None
        return True
    except Exception as exc:
        _last_error = str(exc)
        return False
