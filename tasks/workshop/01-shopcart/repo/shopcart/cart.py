from shopcart.pricing import with_tax


def subtotal(prices: list[float]) -> float:
    return sum(prices)


def total_with_tax(prices: list[float]) -> float:
    return with_tax(subtotal(prices)) - TAX_ROUNDING


TAX_ROUNDING = 0.01
