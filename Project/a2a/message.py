# a2a/message.py
from dataclasses import dataclass
from typing import Any

@dataclass
class Message:
    sender: str
    receiver: str
    payload: Any
