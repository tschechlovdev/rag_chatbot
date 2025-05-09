import json
import os

from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain.schema.messages import message_to_dict, messages_from_dict
from langchain_core.prompt_values import ChatPromptValue
from typing import List
from pathlib import Path

STATE_FILE = Path("conversation.json")

class ConversationManager:
    def save(self, messages: List[BaseMessage]) -> None:
        print(messages)
        print(type(messages[0]))
        with STATE_FILE.open("w") as f:
            json.dump([message_to_dict(m) for m in messages], f)

    def load(self) -> List[BaseMessage]:
        if not STATE_FILE.exists():
            return []
        if STATE_FILE.stat().st_size == 0:
            print("⚠️ Conversation is empty")
            return []
        with STATE_FILE.open() as f:
            message_dict = json.load(f)
            print(message_dict)
            return messages_from_dict(message_dict)
    
    def clear(self) -> None:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
