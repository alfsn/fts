# tests/unit/test_recursive_loader.py

from typing import List

import pytest

from trading_bot.config import ComponentConfig, PluginLoader


# Define dummy classes within the test module so importlib can resolve them
class DummySub:
    """Mock sub-component for testing."""

    def __init__(self, value: int) -> None:
        self.value = value


class DummyParent:
    """Mock parent component that depends on a DummySub object."""

    def __init__(self, sub: DummySub, name: str) -> None:
        self.sub = sub
        self.name = name


class DummyGrandParent:
    """Mock grand-parent component containing nested lists of components."""

    def __init__(self, parents: List[DummyParent], active: bool) -> None:
        self.parents = parents
        self.active = active


def test_recursive_instantiation_simple():
    """
    Tests that a nested ComponentConfig dictionary is recursively
    instantiated and passed as a proper object to the parent class constructor.
    """
    config = ComponentConfig(
        class_path="tests.unit.test_recursive_loader.DummyParent",
        params={
            "name": "TestParent",
            "sub": {
                "class_path": "tests.unit.test_recursive_loader.DummySub",
                "params": {"value": 42},
            },
        },
    )

    # Instantiate via loader
    parent = PluginLoader.instantiate(config)

    # Assertions
    assert isinstance(parent, DummyParent)
    assert parent.name == "TestParent"
    assert isinstance(parent.sub, DummySub)
    assert parent.sub.value == 42


def test_recursive_instantiation_nested_lists():
    """
    Tests that a list containing ComponentConfig dictionaries is resolved
    recursively into a list of active Python instances.
    """
    config = ComponentConfig(
        class_path="tests.unit.test_recursive_loader.DummyGrandParent",
        params={
            "active": True,
            "parents": [
                {
                    "class_path": "tests.unit.test_recursive_loader.DummyParent",
                    "params": {
                        "name": "Parent1",
                        "sub": {
                            "class_path": "tests.unit.test_recursive_loader.DummySub",
                            "params": {"value": 100},
                        },
                    },
                },
                {
                    "class_path": "tests.unit.test_recursive_loader.DummyParent",
                    "params": {
                        "name": "Parent2",
                        "sub": {
                            "class_path": "tests.unit.test_recursive_loader.DummySub",
                            "params": {"value": 200},
                        },
                    },
                },
            ],
        },
    )

    # Instantiate
    grandparent = PluginLoader.instantiate(config)

    # Assertions
    assert isinstance(grandparent, DummyGrandParent)
    assert grandparent.active is True
    assert len(grandparent.parents) == 2

    p1, p2 = grandparent.parents
    assert isinstance(p1, DummyParent)
    assert p1.name == "Parent1"
    assert isinstance(p1.sub, DummySub)
    assert p1.sub.value == 100

    assert isinstance(p2, DummyParent)
    assert p2.name == "Parent2"
    assert isinstance(p2.sub, DummySub)
    assert p2.sub.value == 200
