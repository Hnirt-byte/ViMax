from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse


def source_metadata_path(image_path: str | Path) -> Path:
    return Path(f"{image_path}.source.json")


def is_public_http_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def persist_public_source_url(image_path: str | Path, source_url: str | None) -> None:
    if not is_public_http_url(source_url):
        return
    metadata_path = source_metadata_path(image_path)
    metadata_path.write_text(json.dumps({"source_url": source_url}, ensure_ascii=False, indent=2), encoding="utf-8")


def public_source_url_for_image(image_reference: str | Path) -> str | None:
    reference = str(image_reference)
    if is_public_http_url(reference):
        return reference
    metadata_path = source_metadata_path(reference)
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    source_url = payload.get("source_url") if isinstance(payload, dict) else None
    return source_url if is_public_http_url(source_url) else None
