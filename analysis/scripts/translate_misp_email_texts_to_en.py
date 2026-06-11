"""
Translate MISP email **subject** and **body** using **deep-translator**'s
``GoogleTranslator(source="auto", target=...)`` (same stack as many small apps:
no Google Cloud project, no service account).

Prerequisites
-------------
::

    pip install deep-translator==1.11.4

**Note:** This talks to the same kind of public translation endpoint the library
wraps. Use ``--sleep-between-requests`` to reduce rate-limit issues on large lakes.

``GoogleTranslator`` rejects long strings with ``NotValidLength`` (limit is on the
order of **~5000 characters per request**). Long subject/body are split into
segments of at most ``--chunk-size`` (default 4500), translated separately, then
concatenated. If errors persist, try ``--chunk-size 1800`` (some environments hit
shorter URL limits).

On errors the script **falls back to the original** slice. A **process-local cache**
deduplicates identical strings across emails.

Output format
-------------
``{"meta": {...}, "by_external_id": {"<eid>": {"subject": "...", "body": "..."}}}``

Very long fields are truncated to ``--char-limit`` before chunking; rows may
include ``"body_truncated": true``.

Saved ``subject`` / ``body`` are line-wrapped at ``--store-text-wrap-width`` (default
100) so JSON is readable; use ``0`` to store translator output unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import textwrap
import time
import warnings
from pathlib import Path
from typing import Any

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from misp_email_text_catalog import find_project_root, load_misp_subject_body_by_external_id

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x: Any, **_: Any) -> Any:
        return x


def _wrap_stored_text(text: str, width: int) -> str:
    """Add newlines so long single-line bodies are easier to read in JSON."""
    if width <= 0 or not text:
        return text
    parts: list[str] = []
    for para in text.splitlines():
        if not para.strip():
            parts.append("")
            continue
        parts.append(
            textwrap.fill(
                para,
                width=width,
                break_long_words=True,
                replace_whitespace=False,
            )
        )
    return "\n".join(parts)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _text_chunks(s: str, chunk_size: int) -> list[str]:
    if chunk_size < 64:
        chunk_size = 64
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]


def translate_plain(
    text: str,
    *,
    translator_factory: Any,
    target_language: str,
    text_cache: dict[str, str],
    char_limit: int,
    chunk_size: int,
    sleep_between_requests: float,
    max_retries: int,
) -> str:
    """
    Empty / whitespace-only input is not sent; returns ''.
    Long text is split into segments of at most ``chunk_size`` (GoogleTranslator /
    ``NotValidLength`` limit ~5000). On persistent failure for a segment, that
    segment falls back to the original slice.
    """
    raw = text or ""
    if not raw.strip():
        return ""

    chunk = raw if len(raw) <= char_limit else raw[:char_limit]
    full_ck = _cache_key(chunk)
    if full_ck in text_cache:
        return text_cache[full_ck]

    translator = translator_factory(target_language)
    pieces = _text_chunks(chunk, chunk_size)
    outs: list[str] = []
    for p in pieces:
        if not p:
            continue
        pk = _cache_key(p)
        if pk in text_cache:
            outs.append(text_cache[pk])
            continue
        last_err: BaseException | None = None
        out = p
        for attempt in range(max_retries):
            try:
                out = translator.translate(p)
                time.sleep(sleep_between_requests)
                if not (out or "").strip():
                    out = p
                break
            except BaseException as e:
                last_err = e
                time.sleep(min(60.0, 2.0**attempt))
        else:
            hint = ""
            if last_err is not None and "NotValidLength" in type(last_err).__name__:
                hint = " Try reducing --chunk-size (e.g. 1800)."
            warnings.warn(
                f"translate failed after {max_retries} attempt(s) for one segment; "
                f"using original segment. Last error: {last_err!r}.{hint}"
            )
            out = p
        text_cache[pk] = out
        outs.append(out)

    result = "".join(outs)
    text_cache[full_ck] = result
    return result


def _translator_factory():
    from deep_translator import GoogleTranslator

    def factory(target_language: str) -> GoogleTranslator:
        return GoogleTranslator(source="auto", target=target_language)

    return factory


def run(
    *,
    project_root: Path,
    misp_input: Path,
    json_output: Path,
    target_language: str,
    char_limit: int,
    chunk_size: int,
    sleep_between_requests: float,
    max_retries: int,
    resume: bool,
    max_emails: int | None,
    checkpoint_every: int,
    store_text_wrap_width: int,
) -> None:
    translator_factory = _translator_factory()
    text_cache: dict[str, str] = {}

    raw = load_misp_subject_body_by_external_id(misp_input, project_root=project_root)
    items = list(raw.items())
    if max_emails is not None:
        items = items[: max(0, int(max_emails))]

    by_out: dict[str, dict[str, Any]] = {}
    meta_in: dict[str, Any] = {}
    if resume and json_output.is_file():
        prev = json.loads(json_output.read_text(encoding="utf-8"))
        meta_in = prev.get("meta") if isinstance(prev.get("meta"), dict) else {}
        prev_by = prev.get("by_external_id")
        if isinstance(prev_by, dict):
            for eid, row in prev_by.items():
                if isinstance(row, dict) and str(eid).strip():
                    by_out[str(eid).strip()] = dict(row)

    if max_emails is not None and by_out:
        allowed = {eid for eid, _ in items}
        by_out = {k: v for k, v in by_out.items() if k in allowed}

    to_do: list[tuple[str, str, str]] = [
        (eid, str(rec.get("subject") or ""), str(rec.get("body") or ""))
        for eid, rec in items
        if eid not in by_out
    ]

    def build_payload(note: str | None = None) -> dict[str, Any]:
        existing: dict[str, dict[str, Any]] = {}
        if json_output.is_file():
            try:
                prev = json.loads(json_output.read_text(encoding="utf-8"))
                eb = prev.get("by_external_id")
                if isinstance(eb, dict):
                    for k, v in eb.items():
                        if isinstance(v, dict) and str(k).strip():
                            existing[str(k).strip()] = dict(v)
            except (json.JSONDecodeError, OSError):
                pass
        merged: dict[str, dict[str, Any]] = {**existing, **by_out}
        partial = note == "checkpoint_partial"
        final_by: dict[str, dict[str, Any]] = {}
        for eid, _rec in items:
            if eid in merged:
                final_by[eid] = merged[eid]
            elif not partial:
                final_by[eid] = {"subject": "", "body": ""}
        meta: dict[str, Any] = {
            "source_misp_path": str(misp_input.resolve()),
            "target_language": target_language,
            "api": "deep_translator.GoogleTranslator(source=auto)",
            "n_emails": len(final_by),
            "char_limit_per_field": char_limit,
            "chunk_size_per_translate_call": chunk_size,
            "store_text_wrap_width": store_text_wrap_width,
            "sleep_between_requests": sleep_between_requests,
            "max_retries": max_retries,
            "resume": resume,
            "prior_meta_merged": meta_in,
        }
        if note:
            meta["note"] = note
        return {"meta": meta, "by_external_id": final_by}

    if not to_do:
        _atomic_write_json(
            json_output,
            build_payload("no_new_emails_to_translate"),
        )
        return

    done_since_ckpt = 0
    for eid, subj, body in tqdm(to_do, desc="translate"):
        s = str(subj or "")
        b = str(body or "")
        st = s[:char_limit] if len(s) > char_limit else s
        bt = b[:char_limit] if len(b) > char_limit else b
        flags: dict[str, bool] = {}
        if len(s) > char_limit:
            flags["subject_truncated"] = True
        if len(b) > char_limit:
            flags["body_truncated"] = True

        subj_en = translate_plain(
            st,
            translator_factory=translator_factory,
            target_language=target_language,
            text_cache=text_cache,
            char_limit=char_limit,
            chunk_size=chunk_size,
            sleep_between_requests=sleep_between_requests,
            max_retries=max_retries,
        )
        body_en = translate_plain(
            bt,
            translator_factory=translator_factory,
            target_language=target_language,
            text_cache=text_cache,
            char_limit=char_limit,
            chunk_size=chunk_size,
            sleep_between_requests=sleep_between_requests,
            max_retries=max_retries,
        )

        row: dict[str, Any] = {
            "subject": _wrap_stored_text(subj_en, store_text_wrap_width),
            "body": _wrap_stored_text(body_en, store_text_wrap_width),
        }
        if flags:
            row.update(flags)
        by_out[eid] = row

        done_since_ckpt += 1
        if checkpoint_every > 0 and done_since_ckpt >= checkpoint_every:
            _atomic_write_json(json_output, build_payload("checkpoint_partial"))
            done_since_ckpt = 0

    _atomic_write_json(json_output, build_payload())


def parse_args() -> argparse.Namespace:
    root = find_project_root()
    default_in = root / "data" / "misp" / "incidents-lake-misp.json"
    default_out = root / "data" / "misp" / "incidents-lake-misp-text-en.by_external_id.json"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=default_in, help="MISP lake JSON path.")
    p.add_argument(
        "--output",
        type=Path,
        default=default_out,
        help="Output JSON path (by_external_id only).",
    )
    p.add_argument(
        "--target-language",
        type=str,
        default="en",
        help="Target language code for GoogleTranslator (default en).",
    )
    p.add_argument(
        "--char-limit",
        type=int,
        default=500_000,
        help="Max characters taken from each raw subject/body before chunking (safety cap).",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=4500,
        help=(
            "Max characters per translate() call. GoogleTranslator enforces ~5000; "
            "if you still see NotValidLength, try 1800."
        ),
    )
    p.add_argument(
        "--sleep-between-requests",
        type=float,
        default=0.15,
        help="Seconds to sleep after each successful translate() (rate limiting).",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Retries per chunk on transient errors before falling back to original text for that chunk.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Merge with existing output file and skip eids already present.",
    )
    p.add_argument(
        "--max-emails",
        type=int,
        default=None,
        help="Translate at most this many emails (debug / partial run).",
    )
    p.add_argument(
        "--store-text-wrap-width",
        type=int,
        default=100,
        help="Wrap long lines in saved subject/body (0 = leave as translator returned).",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="If >0, rewrite output JSON after this many newly translated emails (crash safety).",
    )
    return p.parse_args()
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"MISP input not found: {args.input}")
    run(
        project_root=find_project_root(),
        misp_input=args.input,
        json_output=args.output,
        target_language=args.target_language,
        char_limit=args.char_limit,
        chunk_size=args.chunk_size,
        sleep_between_requests=args.sleep_between_requests,
        max_retries=args.max_retries,
        resume=args.resume,
        max_emails=args.max_emails,
        checkpoint_every=args.checkpoint_every,
        store_text_wrap_width=args.store_text_wrap_width,
    )
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
