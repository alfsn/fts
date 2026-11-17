from pathlib import Path

import yaml
from quant_core.models import Portfolio

from .base import BrokerConnector


class LocalYamlConnector(BrokerConnector):
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    @property
    def name(self) -> str:
        return "Local YAML"

    def get_portfolio(self) -> Portfolio:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Portfolio YAML file not found: {self.file_path}")

        with open(self.file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"YAML file {self.file_path} is empty or invalid.")

        return Portfolio.model_validate(data)
