from billing.invoice import invoice_total, line_total


def test_line_total():
    assert line_total(2.0, 5) == 10.0


def test_invoice_total_without_discount():
    assert invoice_total(2.0, 5) == 10.0


def test_invoice_total_applies_bulk_discount():
    assert invoice_total(2.0, 10) == 18.0
