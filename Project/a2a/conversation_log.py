# a2a/conversation_log.py
class ConversationLog:
    def __init__(self):
        self.entries = []

    def add(self, text: str):
        self.entries.append(text)
