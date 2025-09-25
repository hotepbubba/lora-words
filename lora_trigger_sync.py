"""Utilities for syncing local LoRA trigger words from CivitAI metadata.

This module exposes a small command line interface that scans a directory of
LoRA files (``.safetensors`` or ``.ckpt``) and persists any trigger words that
can be recovered for each file. The script first tries to read the metadata
embedded inside the ``.safetensors`` file and falls back to the public CivitAI
API, which can be accessed anonymously or with a personal access token.

Example usage::

    python lora_trigger_sync.py /path/to/lora/dir --database trigger_words.json

"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import datetime as _dt

try:
    import requests
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("The 'requests' package is required to run this script") from exc

try:
    from safetensors import safe_open  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    safe_open = None  # type: ignore


LOGGER = logging.getLogger("lora_trigger_sync")
DEFAULT_EXTENSIONS = (".safetensors", ".ckpt", ".pt")
CIVITAI_HASH_URL = "https://civitai.com/api/v1/model-versions/by-hash/{hash}"  # noqa: E501


class TriggerWordStore:
    """Small helper around the JSON trigger word database."""

    def __init__(self, db_path: Path) -> None:
        self.path = db_path
        self.data: Dict[str, Dict[str, object]] = {}
        if db_path.exists():
            try:
                self.data = json.loads(db_path.read_text("utf-8"))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Unable to parse database {db_path}: {exc}") from exc

    def get(self, model_hash: str) -> Optional[Dict[str, object]]:
        return self.data.get(model_hash)

    def update(
        self,
        model_hash: str,
        *,
        files: Iterable[str],
        trigger_words: Sequence[str],
        source: str,
        model_version_id: Optional[int] = None,
        model_id: Optional[int] = None,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        payload = self.data.get(model_hash, {})
        payload.setdefault("files", [])
        payload["files"] = sorted(set(payload["files"]) | set(files))  # type: ignore[index]
        payload["trigger_words"] = sorted(set(trigger_words))
        payload["source"] = source
        payload["updated_at"] = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        if model_version_id is not None:
            payload["model_version_id"] = model_version_id
        if model_id is not None:
            payload["model_id"] = model_id
        if extra:
            payload.setdefault("extra", {})
            extra_payload = payload["extra"]  # type: ignore[index]
            extra_payload.update(extra)
        self.data[model_hash] = payload

    def sync(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n", "utf-8")


def compute_sha256(path: Path) -> str:
    """Return the hexadecimal SHA256 digest for ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_from_safetensors(path: Path) -> Tuple[List[str], Optional[int], Dict[str, object]]:
    """Try to recover trigger words and model metadata from a safetensors file."""
    if safe_open is None:
        LOGGER.debug("safetensors is not available; skipping metadata extraction for %s", path)
        return [], None, {}

    try:
        with safe_open(path.as_posix(), framework="pt", device="cpu") as handle:  # type: ignore[arg-type]
            metadata = handle.metadata() or {}
    except Exception as exc:  # pragma: no cover - pass through errors
        LOGGER.warning("Failed to read safetensors metadata for %s: %s", path, exc)
        return [], None, {}

    trigger_words: set[str] = set()
    model_version_id: Optional[int] = None
    extra: Dict[str, object] = {}

    for key, value in metadata.items():
        lower_key = key.lower()
        if isinstance(value, str):
            value_str = value.strip()
        else:
            value_str = str(value)

        if value_str:
            # Try JSON payloads first
            parsed = _maybe_parse_json(value_str)
            if isinstance(parsed, dict):
                trigger_words.update(_extract_words_from_dict(parsed))
                model_version_id = model_version_id or _int_or_none(parsed.get("modelVersionId"))
                if "modelId" in parsed:
                    extra.setdefault("model_id", _int_or_none(parsed.get("modelId")))
            elif isinstance(parsed, list):
                trigger_words.update(_stringify_list(parsed))

        if "trained" in lower_key or "trigger" in lower_key:
            trigger_words.update(_split_words(value_str))

    return sorted(trigger_words), model_version_id, extra


def _int_or_none(value: object) -> Optional[int]:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _maybe_parse_json(value: str) -> object:
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def _extract_words_from_dict(payload: Dict[str, object]) -> List[str]:
    words: List[str] = []
    for key in ("trainedWords", "triggerWords", "trigger_words", "trained_words"):
        if key in payload and isinstance(payload[key], list):
            words.extend(_stringify_list(payload[key]))
    return words


def _stringify_list(items: Sequence[object]) -> List[str]:
    return [str(item).strip() for item in items if str(item).strip()]


def _split_words(value: str) -> List[str]:
    separators = [",", "\n", "\t", "|"]
    for separator in separators:
        value = value.replace(separator, " ")
    return [part.strip() for part in value.split() if part.strip()]


def fetch_from_civitai(hash_value: str, token: Optional[str], timeout: int = 30) -> Tuple[List[str], Optional[int], Optional[int], Dict[str, object]]:
    """Query the CivitAI API for trigger words using a model hash."""
    url = CIVITAI_HASH_URL.format(hash=hash_value)
    headers = {"User-Agent": "lora-trigger-sync/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    LOGGER.debug("Querying %s", url)
    response = requests.get(url, headers=headers, timeout=timeout)
    if response.status_code == 404:
        LOGGER.info("No CivitAI entry for hash %s", hash_value)
        return [], None, None, {}
    response.raise_for_status()

    payload = response.json()
    trigger_words = _stringify_list(payload.get("trainedWords", []))
    model_version_id = _int_or_none(payload.get("id"))
    model_id = _int_or_none(payload.get("modelId"))
    return trigger_words, model_version_id, model_id, payload


def iter_lora_files(base_path: Path, extensions: Sequence[str]) -> Iterable[Path]:
    for root, _, files in os.walk(base_path):
        for file_name in files:
            if file_name.startswith('.'):
                continue
            candidate = Path(root, file_name)
            if candidate.suffix.lower() in extensions:
                yield candidate


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync trigger words for local LoRA files")
    parser.add_argument(
        "models_dir",
        type=Path,
        help="Directory containing LoRA model files (recursively scanned).",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("trigger_words.json"),
        help="Path to the JSON database file (default: trigger_words.json in the current directory).",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=list(DEFAULT_EXTENSIONS),
        help="File extensions to scan (default: .safetensors .ckpt .pt)",
    )
    parser.add_argument(
        "--api-token",
        help="Optional CivitAI API token for authenticated requests.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force refresh entries even if they already exist in the database.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level (default: INFO)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s: %(message)s")

    models_dir: Path = args.models_dir
    if not models_dir.exists() or not models_dir.is_dir():
        LOGGER.error("The models directory %s does not exist or is not a directory", models_dir)
        return 2

    store = TriggerWordStore(args.database)
    processed = 0
    updated = 0

    for file_path in iter_lora_files(models_dir, tuple(ext.lower() for ext in args.extensions)):
        processed += 1
        rel_path = file_path.relative_to(models_dir)
        LOGGER.info("Processing %s", rel_path)
        model_hash = compute_sha256(file_path)

        entry = store.get(model_hash)
        if entry and not args.force:
            LOGGER.debug("Hash %s already present, skipping", model_hash)
            continue

        trigger_words, model_version_id, extra = [], None, {}
        if file_path.suffix.lower() == ".safetensors":
            trigger_words, model_version_id, extra = extract_from_safetensors(file_path)

        source = "metadata" if trigger_words else "civitai_api"
        model_id = None

        if not trigger_words:
            trigger_words, model_version_id, model_id, payload = fetch_from_civitai(model_hash, args.api_token)
            extra.setdefault("api_payload", payload)
        else:
            model_id = _int_or_none(extra.get("model_id")) if extra else None
            extra = {k: v for k, v in extra.items() if k != "model_id"}

        if not trigger_words:
            LOGGER.warning("No trigger words found for %s (hash=%s)", rel_path, model_hash)
            continue

        store.update(
            model_hash,
            files=[rel_path.as_posix()],
            trigger_words=trigger_words,
            source=source,
            model_version_id=model_version_id,
            model_id=model_id,
            extra=extra if extra else None,
        )
        updated += 1

    if updated:
        store.sync()
        LOGGER.info("Database updated with %d entries", updated)
    else:
        LOGGER.info("No new trigger words discovered (processed %d files)", processed)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
