import pytest
from unittest.mock import MagicMock, patch
from src.memory.memory_manager import MemoryManager
from src.memory.file_memory_backend import FileMemoryBackend # Import for type hinting/fixture usage
from typing import Dict, Any
import os

@pytest.fixture
def mock_backend():
    """Provides a mock FileMemoryBackend instance."""
    backend = MagicMock(spec=FileMemoryBackend)
    backend.load.return_value = {}  # Default to returning empty memory
    return backend

@pytest.fixture
def memory_manager(mock_backend):
    """Provides a MemoryManager instance with a mocked backend."""
    return MemoryManager(backend=mock_backend)

def test_memory_manager_retrieve_empty(memory_manager, mock_backend):
    thread_id = "test_thread_empty"
    memory = memory_manager.retrieve(thread_id)
    assert memory == {}
    mock_backend.load.assert_called_once_with(thread_id)

def test_memory_manager_retrieve_existing(memory_manager, mock_backend):
    thread_id = "test_thread_existing"
    expected_memory = {"history": ["hello"], "context": {"user": "test"}}
    mock_backend.load.return_value = expected_memory

    memory = memory_manager.retrieve(thread_id)
    assert memory == expected_memory
    mock_backend.load.assert_called_once_with(thread_id)

def test_memory_manager_update(memory_manager, mock_backend):
    thread_id = "test_thread_update"
    new_state = {"history": ["hi", "there"], "context": {"agent": "ai"}}
    memory_manager.update(thread_id, new_state)
    mock_backend.save.assert_called_once_with(thread_id, new_state)

def test_memory_manager_clear(memory_manager, mock_backend):
    thread_id = "test_thread_clear"
    memory_manager.clear(thread_id)
    mock_backend.delete.assert_called_once_with(thread_id)

def test_memory_manager_default_backend_creation():
    # Ensure default backend is created if none is provided
    with patch('src.memory.memory_manager.FileMemoryBackend') as MockFileMemoryBackend:
        manager = MemoryManager()
        MockFileMemoryBackend.assert_called_once()
        
def test_file_memory_backend_lifecycle(tmp_path):
    # This is an integration test for FileMemoryBackend itself
    # It will create actual files in a temporary directory
    storage_dir = tmp_path / "test_memory_store"
    backend = FileMemoryBackend(storage_dir=str(storage_dir))

    thread_id = "integration_test_thread"
    test_state = {"data": "some value", "list": [1, 2, 3]}

    # Test save
    backend.save(thread_id, test_state)
    file_path = os.path.join(storage_dir, f"{thread_id}.json")
    assert os.path.exists(file_path)

    # Test load
    loaded_state = backend.load(thread_id)
    assert loaded_state == test_state

    # Test update (save over existing)
    updated_state = {"data": "new value", "list": [4, 5, 6]}
    backend.save(thread_id, updated_state)
    loaded_updated_state = backend.load(thread_id)
    assert loaded_updated_state == updated_state

    # Test delete
    backend.delete(thread_id)
    assert not os.path.exists(file_path)

    # Test load non-existent
    non_existent_memory = backend.load("non_existent_thread")
    assert non_existent_memory == {}
