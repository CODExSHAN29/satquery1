from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseTool(ABC):
    """
    Abstract Base Class for all SatQuery AI specialist tools.

    Every concrete tool must implement :meth:`execute` (the canonical entry
    point) and expose a human-readable :attr:`tool_name` / :attr:`description`.
    :meth:`run` is provided as a thin alias so callers may use either name.
    """

    @property
    @abstractmethod
    def tool_name(self) -> str:
        """Short machine-readable identifier for the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        pass

    @abstractmethod
    def execute(
        self,
        query: str = "",
        image_path: str = "",
        images: Optional[List[str]] = None,
        task_mode: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run the tool against a natural-language query and one or more
        satellite image paths. Returns a structured result dictionary.
        """
        pass

    def run(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convenience alias for :meth:`execute`."""
        return self.execute(*args, **kwargs)