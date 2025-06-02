import json
import os

from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain.schema.messages import message_to_dict, messages_from_dict
from langchain_core.prompt_values import ChatPromptValue
from typing import List
from pathlib import Path


class ConversationManager:
    """
    A class to manage conversations by saving, loading, and clearing messages.

    Attributes:
        STATE_FILE (str): The file path to store conversation messages.

    Methods:
        save(messages: List[BaseMessage]) -> None:
            Saves a list of BaseMessage objects to a JSON file.

        load() -> List[BaseMessage]:
            Loads BaseMessage objects from a JSON file. Returns an empty list if the file is empty or does not exist.

        clear() -> None:
            Deletes the JSON file containing conversation messages.
    """

    def __init__(self, state_file: str = "conversation.json"):
        """
        Initializes the ConversationManager and sets the state file path.

        Parameters:
        state_file (str, optional): The path to the state file. Defaults to "conversation.json".

        Attributes:
        self.STATE_FILE (Path): The state file path.
        """
        self.STATE_FILE = Path(state_file)


    def save(self, messages: List[BaseMessage]) -> None:
        """
        Saves a list of BaseMessage objects to a JSON file.

        Args:
            messages (List[BaseMessage]): A list of BaseMessage objects to be saved.

        Returns:
            None
        """
        with self.STATE_FILE.open("w") as f:
            json.dump([message_to_dict(m) for m in messages], f)

    def load(self) -> List[BaseMessage]:
        """
        Loads BaseMessage objects from a JSON file. Returns an empty list if the file is empty or does not exist.

        Returns:
            List[BaseMessage]: A list of BaseMessage objects loaded from the JSON file.
        """
        if not self.STATE_FILE.exists():
            return []
        if self.STATE_FILE.stat().st_size == 0:
            print("Conversation is empty")
            return []
        with self.STATE_FILE.open() as f:
            message_dict = json.load(f)
            print(message_dict)
            return messages_from_dict(message_dict)
    
    def clear(self) -> None:
        """
        Deletes the JSON file containing conversation messages.

        Returns:
            None
        """
        if self.STATE_FILE.exists():
            self.STATE_FILE.unlink()
