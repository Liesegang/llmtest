from abc import ABC, abstractmethod

class TTSInterface(ABC):
    @abstractmethod
    def speak(self, text: str, **kwargs):
        """
        Convert text to speech and output via AudioIO.
        """
        pass
