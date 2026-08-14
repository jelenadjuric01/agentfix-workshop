from config.tokens import clean, is_comment


def parse(text: str) -> dict[str, str]:
    settings: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = clean(raw_line)
        if not line or is_comment(line):
            continue
        key, _, value = line.partition("=")
        settings[clean(key)] = clean(value)
    return settings
