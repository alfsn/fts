# src/trading_bot/core/loader.py

import importlib
from typing import Any, Type

from ..config import ComponentConfig


class PluginLoader:
    """Utility to dynamically load and instantiate components."""

    @staticmethod
    def load_class(class_path: str) -> Type[Any]:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    @classmethod
    def _resolve_params(cls, val: Any) -> Any:
        """Recursively checks and instantiates nested configurations."""
        if isinstance(val, dict):
            if "class_path" in val:
                # Resolve inner params first
                inner_params = cls._resolve_params(val.get("params", {}))
                resolved_config = ComponentConfig(
                    class_path=val["class_path"], params=inner_params
                )
                return cls.instantiate(resolved_config)
            else:
                return {k: cls._resolve_params(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [cls._resolve_params(item) for item in val]
        return val

    @classmethod
    def instantiate(cls, config: ComponentConfig) -> Any:
        klass = cls.load_class(config.class_path)
        # Recursively resolve any nested component configs in parameters
        resolved_params = {k: cls._resolve_params(v) for k, v in config.params.items()}
        return klass(**resolved_params)
