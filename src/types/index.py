from typing import Any, Dict, Protocol

class TopicData(Protocol):
    topic_name: str
    data: Any
    timestamp: float

    def get_data(self) -> Any:
        pass

    def get_timestamp(self) -> float:
        pass

    def get_topic_name(self) -> str:
        pass