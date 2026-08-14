from billing.discounts import bulk_discount


def line_total(unit_price: float, quantity: int) -> float:
    return unit_price * quantity


def invoice_total(unit_price: float, quantity: int) -> float:
    gross = line_total(unit_price, quantity)
    return gross - bulk_discount(quantity, gross)
