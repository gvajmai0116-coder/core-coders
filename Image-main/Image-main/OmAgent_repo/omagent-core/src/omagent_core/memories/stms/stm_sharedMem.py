import os
from typing import Any
from omagent_core.memories.stms.stm_base import STMBase, WorkflowInstanceProxy
from omagent_core.utils.registry import registry

@registry.register_component()
class SharedMemSTM(STMBase):
    """In-memory Short-Term Memory (STM) implementation for local workflow execution.
    Avoids multiprocessing shared memory serialization corruption issues.
    """
    _global_store = {}

    def __init__(self, id=None):
        super().__init__()
        self.id = id

    def __call__(self, workflow_instance_id: str):
        return WorkflowInstanceProxy(self, workflow_instance_id)

    def __getitem__(self, key: tuple | str) -> Any:
        if isinstance(key, tuple):
            workflow_instance_id, key = key
            if workflow_instance_id not in self._global_store:
                raise KeyError(key)
            if key not in self._global_store[workflow_instance_id]:
                raise KeyError(key)
            return self._global_store[workflow_instance_id][key]
        else:
            workflow_instance_id = key
            return self._global_store.get(workflow_instance_id, {})

    def __setitem__(self, key: tuple, value: Any) -> None:
        workflow_instance_id, key = key
        if workflow_instance_id not in self._global_store:
            self._global_store[workflow_instance_id] = {}
        self._global_store[workflow_instance_id][key] = value

    def __delitem__(self, workflow_instance_id: str, key: str) -> None:
        if workflow_instance_id in self._global_store and key in self._global_store[workflow_instance_id]:
            del self._global_store[workflow_instance_id][key]
        else:
            raise KeyError(key)

    def __contains__(self, key: tuple) -> bool:
        workflow_instance_id, key = key
        return workflow_instance_id in self._global_store and key in self._global_store[workflow_instance_id]

    def keys(self, workflow_instance_id: str) -> list:
        return list(self._global_store.get(workflow_instance_id, {}).keys())

    def values(self, workflow_instance_id: str) -> list:
        return list(self._global_store.get(workflow_instance_id, {}).values())

    def items(self, workflow_instance_id: str) -> list:
        return list(self._global_store.get(workflow_instance_id, {}).items())

    def get(self, workflow_instance_id: str, key: str, default: Any = None) -> Any:
        try:
            return self[workflow_instance_id, key]
        except KeyError:
            return default

    def clear(self, workflow_instance_id: str) -> None:
        if workflow_instance_id in self._global_store:
            self._global_store[workflow_instance_id].clear()

    def pop(self, workflow_instance_id: str, key: str, default: Any = None) -> Any:
        if workflow_instance_id in self._global_store and key in self._global_store[workflow_instance_id]:
            return self._global_store[workflow_instance_id].pop(key)
        return default

    def update(self, workflow_instance_id: str, other: dict) -> None:
        if workflow_instance_id not in self._global_store:
            self._global_store[workflow_instance_id] = {}
        self._global_store[workflow_instance_id].update(other)

    def __len__(self, workflow_instance_id: str) -> int:
        return len(self._global_store.get(workflow_instance_id, {}))

    def cleanup(self):
        pass
