"""Per-customer API keys and usage for a hosted deployment.

Keys live in a dict-like store (a plain dict in tests, a modal.Dict on Modal), so they can be
added and revoked without a redeploy. Only a SHA-256 hash of each key is stored:
    keys[sha256(key)] = customer name
Usage is one entry per finished task, so a retried write never double counts:
    usage[task_id] = {"owner": ..., "pages": ..., "finished_at": ...}
"""
import hashlib
import re
import secrets
from collections.abc import Mapping, MutableMapping

DEFAULT_OWNER = "default"  # owner of tasks made with the self-host key (and of older tasks)
NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def add_key(keys: MutableMapping, customer: str) -> str:
    """Make a new key for `customer`, store its hash and return the key (shown once)."""
    if not NAME.match(customer) or customer == DEFAULT_OWNER:
        raise ValueError(f"customer name must be lowercase letters, digits, '-' or '_' "
                         f"(and not '{DEFAULT_OWNER}'): {customer!r}")
    key = "dp_" + secrets.token_urlsafe(32)
    keys[hash_key(key)] = customer
    return key


def customer_for(keys: Mapping, key: str) -> str | None:
    return keys.get(hash_key(key)) if key else None


def list_customers(keys: Mapping) -> dict[str, int]:
    """Customer name -> number of active keys."""
    out: dict[str, int] = {}
    for customer in keys.values():
        out[customer] = out.get(customer, 0) + 1
    return dict(sorted(out.items()))


def revoke(keys: MutableMapping, customer: str) -> int:
    """Remove every key of `customer`; returns how many were removed."""
    gone = [h for h, c in list(keys.items()) if c == customer]
    for h in gone:
        keys.pop(h, None)
    return len(gone)


def record_usage(usage: MutableMapping, rec: Mapping) -> None:
    """Record the pages of a succeeded task against its owner."""
    if rec.get("status") != "Succeeded" or not rec.get("page_count"):
        return
    usage[rec["task_id"]] = {"owner": rec.get("owner") or DEFAULT_OWNER, "pages": int(rec["page_count"]),
                             "finished_at": rec.get("finished_at")}


def usage_report(usage: Mapping) -> dict[str, dict[str, int]]:
    """Customer name -> {"tasks": n, "pages": n}."""
    out: dict[str, dict[str, int]] = {}
    for u in usage.values():
        row = out.setdefault(u["owner"], {"tasks": 0, "pages": 0})
        row["tasks"] += 1
        row["pages"] += u["pages"]
    return dict(sorted(out.items()))
