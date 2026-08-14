from shopcart.cart import subtotal, total_with_tax


def test_subtotal():
    assert subtotal([1.0, 2.0, 3.0]) == 6.0


def test_total_with_tax():
    assert total_with_tax([10.0]) == 12.0
