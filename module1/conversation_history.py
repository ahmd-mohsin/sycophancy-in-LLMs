import json
import os
from typing import Optional


class ConversationHistoryManager:

    def __init__(self, history_path: Optional[str] = None):
        self.history_path = history_path or "conversation_history.json"
        self._history = {}
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "r") as f:
                    self._history = json.load(f)
            except Exception:
                self._history = {}

    def add_turn(self, session_id: str, user_msg: str, assistant_msg: str):
        if session_id not in self._history:
            self._history[session_id] = []
        self._history[session_id].append((user_msg, assistant_msg))
        self._save()

    def get_turns(self, session_id: str) -> list:
        return self._history.get(session_id, [])

    def _save(self):
        try:
            with open(self.history_path, "w") as f:
                json.dump(self._history, f, indent=2)
        except Exception:
            pass