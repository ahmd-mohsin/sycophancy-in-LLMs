import json
import time
from pathlib import Path
from typing import Optional

from config import CONVERSATION_HISTORY_PATH, MAX_HISTORY_TURNS


class ConversationHistoryManager:

    def __init__(self, history_path: Optional[str] = None):
        self.history_path = Path(history_path or CONVERSATION_HISTORY_PATH)
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not self.history_path.exists() or self.history_path.stat().st_size == 0:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_history({"sessions": {}})

    def _read_history(self) -> dict:
        with open(self.history_path, "r") as f:
            return json.load(f)

    def _write_history(self, data: dict):
        with open(self.history_path, "w") as f:
            json.dump(data, f, indent=2)

    def create_session(self, session_id: str) -> str:
        history = self._read_history()
        if session_id not in history["sessions"]:
            history["sessions"][session_id] = {
                "created_at": time.time(),
                "turns": [],
            }
            self._write_history(history)
        return session_id

    def add_turn(self, session_id: str, user_message: str, assistant_response: str):
        history = self._read_history()
        if session_id not in history["sessions"]:
            self.create_session(session_id)
            history = self._read_history()

        turn = {
            "timestamp": time.time(),
            "user": user_message,
            "assistant": assistant_response,
        }
        history["sessions"][session_id]["turns"].append(turn)

        if len(history["sessions"][session_id]["turns"]) > MAX_HISTORY_TURNS:
            history["sessions"][session_id]["turns"] = history["sessions"][session_id][
                "turns"
            ][-MAX_HISTORY_TURNS:]

        self._write_history(history)

    def get_turns(self, session_id: str) -> list[tuple[str, str]]:
        history = self._read_history()
        if session_id not in history["sessions"]:
            return []
        return [
            (t["user"], t["assistant"])
            for t in history["sessions"][session_id]["turns"]
        ]

    def get_last_n_turns(self, session_id: str, n: int) -> list[tuple[str, str]]:
        turns = self.get_turns(session_id)
        return turns[-n:] if len(turns) > n else turns

    def clear_session(self, session_id: str):
        history = self._read_history()
        if session_id in history["sessions"]:
            del history["sessions"][session_id]
            self._write_history(history)

    def session_exists(self, session_id: str) -> bool:
        history = self._read_history()
        return session_id in history["sessions"]

    def format_history_for_prompt(self, session_id: str, max_turns: int = 10) -> str:
        turns = self.get_last_n_turns(session_id, max_turns)
        if not turns:
            return ""
        lines = []
        for i, (user_msg, asst_msg) in enumerate(turns, 1):
            lines.append(f"Turn {i}:")
            lines.append(f"  User: {user_msg}")
            lines.append(f"  Assistant: {asst_msg}")
        return "\n".join(lines)