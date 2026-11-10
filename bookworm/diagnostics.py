"""Public errors omit credential-bearing upstream URLs and response bodies."""
from .errors import BookwormError


def describe_error(exc: BaseException) -> tuple[str, str, int]:
    import httpx
    from openai import APIError
    from pydantic import ValidationError

    if isinstance(exc, BookwormError):
        return exc.code, str(exc), exc.exit_code
    if isinstance(exc, KeyboardInterrupt):
        return "interrupted", "Interrupted; checkpoints and earlier results are preserved", 130
    if isinstance(exc, ValidationError):
        message = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors(include_input=False))
        return "invalid_input", message, 2
    if isinstance(exc, (httpx.HTTPError, APIError)):
        return "upstream", f"{type(exc).__name__}: external service request failed; check network and API configuration", 69
    if isinstance(exc, FileNotFoundError):
        return "invalid_input", "Input file or directory does not exist", 2
    return "internal", f"{type(exc).__name__}: operation failed; inspect the local run checkpoint", 1
