from config.parser import parse


def test_parse_simple():
    assert parse("a=1\nb=2") == {"a": "1", "b": "2"}


def test_parse_skips_comments():
    assert parse("# ignored\na=1") == {"a": "1"}


def test_parse_trims_whitespace():
    assert parse("  a =  1  ") == {"a": "1"}
