BULK_THRESHOLD = 10
BULK_RATE = 0.1


def bulk_discount(quantity: int, amount: float) -> float:
    if quantity > BULK_THRESHOLD:
        return amount * BULK_RATE
    return 0.0
