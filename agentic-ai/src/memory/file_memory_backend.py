import json
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class FileMemoryBackend:
    """
    A simple file-based persistence backend for memory.
    Stores each thread's memory as a separate JSON file.
    """
    def __init__(self, storage_dir: str = "memory_store"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        logger.info(f"FileMemoryBackend initialized with storage directory: {self.storage_dir}")

    def _get_file_path(self, thread_id: str) -> str:
        """Helper to get the file path for a given thread ID."""
        return os.path.join(self.storage_dir, f"{thread_id}.json")

    def load(self, thread_id: str) -> Dict[str, Any]:
        """Loads memory for a given thread_id from a file."""
        file_path = self._get_file_path(thread_id)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    memory = json.load(f)
                logger.debug(f"Loaded memory for {thread_id} from {file_path}.")
                return memory
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {file_path}: {e}")
                return {}
            except Exception as e:
                logger.error(f"Error loading memory from {file_path}: {e}")
                return {}
        logger.debug(f"No memory file found for {thread_id} at {file_path}.")
        return {}

    def save(self, thread_id: str, state: Dict[str, Any]):
        """Saves memory for a given thread_id to a file."""
        file_path = self._get_file_path(thread_id)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=4)
            logger.debug(f"Saved memory for {thread_id} to {file_path}.")
        except Exception as e:
            logger.error(f"Error saving memory to {file_path}: {e}")

    def delete(self, thread_id: str):
        """Deletes memory for a given thread_id."""
        file_path = self._get_file_path(thread_id)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"Deleted memory for {thread_id} from {file_path}.")
            except Exception as e:
                logger.error(f"Error deleting memory file {file_path}: {e}")
