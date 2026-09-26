"""A spend ceiling the runner cannot exceed.

Browser agents send the whole page state on every step, so token use grows quietly and a batch can
cost several times what you expected. This tracks spend across a batch and stops it, rather than
discovering the overrun on the invoice.

Prices change. `PRICES` is a starting point, not an authority: check your provider's current rates
and override with MODEL_PRICE_IN / MODEL_PRICE_OUT (US dollars per million tokens) in .env.
"""

from __future__ import annotations

import os

# US dollars per million tokens, (input, output). Verify before trusting a number derived from this.
PRICES = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o-mini": (0.15, 0.60),
}
DEFAULT_PRICE = (0.40, 1.60)


def price_for(model: str) -> tuple[float, float]:
    override = (os.environ.get("MODEL_PRICE_IN"), os.environ.get("MODEL_PRICE_OUT"))
    if all(override):
        return float(override[0]), float(override[1])
    for name, price in PRICES.items():
        if model.startswith(name):
            return price
    return DEFAULT_PRICE


class BudgetExceeded(RuntimeError):
    """The batch has spent its ceiling. Stopping is the correct outcome, not an error to retry."""


class Budget:
    """Running total for one batch, in tokens and in estimated dollars."""

    def __init__(self, max_usd: float | None = None, model: str = "gpt-4.1-mini"):
        if max_usd is None:
            max_usd = float(os.environ.get("MAX_SPEND_USD", "5") or 5)
        self.max_usd = max_usd
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0
        self.runs = 0

    def add(self, input_tokens: int = 0, output_tokens: int = 0, total_tokens: int = 0) -> None:
        """Add one run's usage. When only a total is known, charge it all at the input rate and say
        so: under-charging a cap is the failure mode that matters."""
        if total_tokens and not (input_tokens or output_tokens):
            input_tokens = total_tokens
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)
        self.runs += 1

    @property
    def usd(self) -> float:
        price_in, price_out = price_for(self.model)
        return (self.input_tokens * price_in + self.output_tokens * price_out) / 1e6

    @property
    def remaining(self) -> float:
        return self.max_usd - self.usd

    def check(self) -> None:
        if self.usd >= self.max_usd:
            raise BudgetExceeded(
                f"spent about ${self.usd:.2f} of a ${self.max_usd:.2f} ceiling "
                f"over {self.runs} runs ({self.total_tokens:,} tokens). Stopping."
            )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def line(self) -> str:
        return (f"spent about ${self.usd:.2f} of ${self.max_usd:.2f} "
                f"({self.total_tokens:,} tokens over {self.runs} runs)")

    def projected_runs(self) -> float | None:
        """How many more runs the ceiling allows at the current average."""
        if not self.runs or not self.usd:
            return None
        return self.remaining / (self.usd / self.runs)
