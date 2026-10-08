"""Persistent user profile (memory-based personalization). Stored locally in user_profiles/<name>.json."""
import json
import re

from . import config

PROFILE_DIR = config.ROOT / "user_profiles"
ROLES = ["general user", "clinician", "pharmacist / researcher", "student", "marketer / analyst"]
DETAILS = ["brief", "standard", "detailed"]
LANGUAGES = ["English", "Arabic"]


def _path(user: str):
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", user.strip().lower()) or "guest"
    return PROFILE_DIR / f"{safe}.json"


def load_profile(user: str) -> dict:
    base = {"role": ROLES[0], "detail": "standard", "language": "English", "topics": []}
    p = _path(user)
    if p.exists():
        try:
            base.update(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    return base


def save_profile(user: str, profile: dict) -> None:
    PROFILE_DIR.mkdir(exist_ok=True)
    _path(user).write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")


def remember_topic(profile: dict, query: str, keep: int = 8) -> dict:
    topics = [t for t in profile.get("topics", []) if t != query[:100]] + [query[:100]]
    profile["topics"] = topics[-keep:]
    return profile


def profile_prompt(profile: dict | None) -> str:
    """Text injected into the prompt. Personalizes tone/depth/language only; never safety rules."""
    if not profile:
        return ""
    text = (f"USER PROFILE (adapt tone, depth and language only; never relax safety rules): "
            f"role={profile['role']}; preferred detail={profile['detail']}; answer language={profile['language']}.")
    if profile["language"] != "English":
        text += f" Write all JSON values in {profile['language']}; keep JSON keys and file names in English."
    if profile.get("topics"):
        text += " Recent topics of this user: " + " | ".join(profile["topics"][-5:]) + "."
    return text
