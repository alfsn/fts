# src/trading_bot/core/spec_base.py

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field

from trading_bot.config import ROOT_DIR

# Anchor PROJECT_ROOT to the central application ROOT_DIR
PROJECT_ROOT = Path(ROOT_DIR)


def deep_merge(dict_a: Dict[str, Any], dict_b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merges dict_b into dict_a in place.
    If both dict_a and dict_b have a dict at a given key, they are merged recursively.
    Otherwise, the value in dict_b overwrites dict_a.
    """
    for key, value in dict_b.items():
        if key in dict_a and isinstance(dict_a[key], dict) and isinstance(value, dict):
            deep_merge(dict_a[key], value)
        else:
            dict_a[key] = copy.deepcopy(value)
    return dict_a


class BaseComposableSpec(BaseModel):
    """
    Base Pydantic spec class supporting YAML imports, dict deep-merging,
    and root attribute overriding with deterministic root-anchored paths.
    """

    imports: List[str] = Field(
        default_factory=list,
        description="List of relative or absolute paths to component YAML spec files",
    )

    @classmethod
    def _resolve_yaml_dict(
        cls, file_path: Path, visited: Optional[set] = None
    ) -> Dict[str, Any]:
        if visited is None:
            visited = set()

        resolved_path = file_path.resolve()
        if resolved_path in visited:
            raise ValueError(
                f"Circular dependency detected in spec imports: {resolved_path}"
            )
        visited.add(resolved_path)

        if not resolved_path.exists():
            raise FileNotFoundError(f"Spec component file not found: {resolved_path}")

        with open(resolved_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        imports = data.pop("imports", [])
        merged_dict: Dict[str, Any] = {}

        # Resolve imported component files sequentially using deterministic anchors
        for imp in imports:
            imp_path = Path(imp)
            if not imp_path.is_absolute():
                candidates = [
                    resolved_path.parent / imp_path,
                    PROJECT_ROOT / imp_path,
                ]
                found = False
                for candidate in candidates:
                    if candidate.exists():
                        imp_dict = cls._resolve_yaml_dict(candidate, visited)
                        deep_merge(merged_dict, imp_dict)
                        found = True
                        break
                if not found:
                    raise FileNotFoundError(
                        f"Could not resolve imported spec file '{imp}' from '{resolved_path}' or PROJECT_ROOT '{PROJECT_ROOT}'"
                    )
            else:
                imp_dict = cls._resolve_yaml_dict(imp_path, visited)
                deep_merge(merged_dict, imp_dict)

        # Merge current file data over imported components
        deep_merge(merged_dict, data)
        return merged_dict

    @classmethod
    def from_yaml(cls, path: Union[str, Path]):
        """Loads and validates a spec from a YAML specification file."""
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = (PROJECT_ROOT / file_path).resolve()
        data_dict = cls._resolve_yaml_dict(file_path)
        return cls(**data_dict)
