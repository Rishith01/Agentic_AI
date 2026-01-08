# a2a/message_bus.py
class MessageBus:
    def __init__(self, conversation_log=None):
        self.log = []
        self.conversation_log = conversation_log

    def send(self, message):
        self.log.append(message)
        if self.conversation_log:
            self.conversation_log.add(
                f"[{message.sender}] → {message.receiver}"
            )
        return message.payload
