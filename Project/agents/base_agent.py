# agents/base_agent.py
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, name: str, bus):
        self.name = name
        self.bus = bus

    @abstractmethod
    def run(self, input_data):
        pass
