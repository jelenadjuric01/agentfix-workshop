COMMENT_PREFIX = "//"


def is_comment(line: str) -> bool:
    return line.startswith(COMMENT_PREFIX)


def clean(line: str) -> str:
    return line
