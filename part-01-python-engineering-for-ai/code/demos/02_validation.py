"""02_validation.py - the customs desk in action.

What it does: feeds three payloads to ModelReply: a good one, a broken one, and one that Pydantic
repairs by converting "5" to 5. It prints what was accepted and exactly what was rejected.
What it needs: pydantic. Runs on any laptop.
What to expect: one accepted reply, a list of three problems for the broken one, and a converted number.
Run it:  uv run python demos/02_validation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError

from caredesk_fanout.models import ModelReply

GOOD = {"model": "alpha-mini", "text": "All clear.", "usage": {"prompt_tokens": 5, "completion_tokens": 7}}
BROKEN = {"model": "alpha-mini", "text": "", "usage": {"prompt_tokens": -3, "completion_tokens": "twelve"}}
LAX = {"model": "alpha-mini", "text": "All clear.", "usage": {"prompt_tokens": "5", "completion_tokens": 7}}


def check(label: str, payload: dict) -> None:
    """Try to build a ModelReply and report what the desk decided."""
    print(f"--- {label}")
    try:
        reply = ModelReply.model_validate(payload)
        print(f"accepted: total tokens = {reply.usage.total}, prompt_tokens = {reply.usage.prompt_tokens!r}")
    except ValidationError as exc:
        print(f"rejected with {exc.error_count()} problems:")
        for err in exc.errors():
            where = ".".join(str(p) for p in err["loc"])
            print(f"  {where:<26} {err['msg']}")


check("a good parcel", GOOD)
check("a broken parcel", BROKEN)
check('a parcel with "5" where 5 is expected', LAX)
