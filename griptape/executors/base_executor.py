from __future__ import annotations
from typing import TYPE_CHECKING
import inspect
import os
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from griptape.core import BaseTool


class BaseExecutor(ABC):
    def execute(self, tool_action: callable, value: bytes) -> bytes:
        value = self.before_execute(tool_action, value)
        result = self.try_execute(tool_action, value)
        result = self.after_execute(tool_action, result)

        return result

    def before_execute(self, tool_action: callable, value: bytes) -> bytes:
        return value

    def after_execute(self, tool_action: callable, result: bytes) -> bytes:
        for middleware in tool_action.__self__.middleware.get(tool_action.config["name"], []):
            result = middleware.process_output(tool_action, result)

        return result

    @abstractmethod
    def try_execute(self, tool_action: callable, value: bytes) -> bytes:
        ...

    def tool_dir(self, tool: BaseTool):
        class_file = inspect.getfile(tool.__class__)

        return os.path.dirname(os.path.abspath(class_file))
