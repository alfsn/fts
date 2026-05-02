# src/plugins/forecasting/data_providers.py

from datetime import datetime, timezone
from typing import Sequence

from trading_bot.core.schemas import ExternalData
from trading_bot.data_ingestion.abc import (
    BaseExternalDataProvider,
    BaseMarketDataProvider,
)


class CCLProvider(BaseExternalDataProvider):
    """
    Calculates the CCL (Contado con Liquidación) rate using the midpoint
    of a local asset and its ADR.
    """

    def __init__(
        self,
        market_provider: BaseMarketDataProvider,
        local_id: str,
        adr_id: str,
        ratio: float = 10.0,
    ) -> None:
        self.market_provider = market_provider
        self.local_id = local_id
        self.adr_id = adr_id
        self.ratio = ratio  # GGAL is 1 ADR = 10 Local shares

    @property
    def source_name(self) -> str:
        return f"ccl_provider_{self.local_id}"

    def fetch_data(self) -> Sequence[ExternalData]:
        try:
            # 1. Get latest prices (midpoint)
            local_ob = self.market_provider.get_order_book(self.local_id)
            adr_ob = self.market_provider.get_order_book(self.adr_id)

            if (
                not local_ob.bids
                or not local_ob.asks
                or not adr_ob.bids
                or not adr_ob.asks
            ):
                return []

            local_mid = (local_ob.bids[0].price + local_ob.asks[0].price) / 2
            adr_mid = (adr_ob.bids[0].price + adr_ob.asks[0].price) / 2

            # CCL = (Local Price * Ratio) / ADR Price (in USD)
            # Result is ARS/USD
            ccl_rate = (local_mid * self.ratio) / adr_mid

            return [
                ExternalData(
                    source=self.source_name,
                    timestamp=datetime.now(timezone.utc),
                    content={
                        "ccl_rate": ccl_rate,
                        "local_mid": local_mid,
                        "adr_mid": adr_mid,
                    },
                )
            ]
        except Exception:
            return []
