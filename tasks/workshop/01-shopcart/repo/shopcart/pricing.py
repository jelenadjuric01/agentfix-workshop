TAX_RATE = 0.2


def with_tax(amount: float) -> float:
    return amount * (1 + TAX_RATE)
