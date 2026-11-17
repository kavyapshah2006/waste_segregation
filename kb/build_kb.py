"""Validate the knowledge base sources and compile them into kb.sqlite.

Usage:
    uv run python kb/build_kb.py                 # writes kb/build/kb.sqlite
    uv run python kb/build_kb.py --check         # validate only

The SQLite file is what the Android app and the host pipeline read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

KB_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = KB_DIR / "build" / "kb.sqlite"
REQUIRED_LANGS = ("en", "hi")


@dataclass
class KbSource:
    items: dict[str, dict]
    cities: dict[str, dict]
    strings: dict
    files: list[Path] = field(default_factory=list)


def load_source(kb_dir: Path = KB_DIR) -> KbSource:
    items, cities, files = {}, {}, []
    for path in sorted((kb_dir / "items").glob("*.json")):
        items[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        files.append(path)
    for path in sorted((kb_dir / "cities").glob("*.json")):
        cities[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        files.append(path)
    strings_path = kb_dir / "strings.json"
    files.append(strings_path)
    strings = json.loads(strings_path.read_text(encoding="utf-8"))
    return KbSource(items, cities, strings, files)


def _validators(kb_dir: Path) -> tuple[Draft202012Validator, Draft202012Validator]:
    item_schema = json.loads((kb_dir / "schema.json").read_text(encoding="utf-8"))
    city_schema = json.loads((kb_dir / "city_schema.json").read_text(encoding="utf-8"))
    registry = Registry().with_resources(
        [(s["$id"], Resource.from_contents(s)) for s in (item_schema, city_schema)]
    )
    return (
        Draft202012Validator(item_schema, registry=registry),
        Draft202012Validator(city_schema, registry=registry),
    )


def validate(src: KbSource, kb_dir: Path = KB_DIR) -> list[str]:
    """Return a list of human-readable errors; empty means valid."""
    errors: list[str] = []
    item_validator, city_validator = _validators(kb_dir)

    for stem, item in src.items.items():
        for err in item_validator.iter_errors(item):
            errors.append(f"items/{stem}.json: {err.json_path}: {err.message}")
        if item.get("id") != stem:
            errors.append(f"items/{stem}.json: id {item.get('id')!r} does not match file name")
        for n, rule in enumerate(item.get("condition_rules", [])):
            then = rule.get("then", {})
            if "stream" in then and then["stream"] != item.get("stream") and "steps" not in then:
                errors.append(
                    f"items/{stem}.json: condition_rules[{n}] changes the stream but does not "
                    "replace the steps"
                )

    for stem, city in src.cities.items():
        for err in city_validator.iter_errors(city):
            errors.append(f"cities/{stem}.json: {err.json_path}: {err.message}")
        if city.get("id") != stem:
            errors.append(f"cities/{stem}.json: id {city.get('id')!r} does not match file name")
        for item_id in city.get("overrides", {}):
            if item_id not in src.items:
                errors.append(f"cities/{stem}.json: override for unknown item {item_id!r}")
        used = {i.get("drop_off") for i in src.items.values()} - {None}
        used |= {o.get("drop_off") for o in city.get("overrides", {}).values()} - {None}
        for key in sorted(used - set(city.get("drop_off_info", {}))):
            errors.append(f"cities/{stem}.json: missing drop_off_info for {key!r}")
        for colour in city.get("bin_colours", {}).values():
            if colour not in src.strings.get("colours", {}):
                errors.append(f"cities/{stem}.json: colour {colour!r} has no name in strings.json")

    if "default_in" not in src.cities:
        errors.append("cities/default_in.json is required")

    owner: dict[str, str] = {}
    for item_id, item in src.items.items():
        for cls in item.get("detector_classes", []):
            if cls in owner:
                errors.append(f"detector class {cls!r} used by both {owner[cls]} and {item_id}")
            owner[cls] = item_id

    for section in ("streams", "colours"):
        for key, text in src.strings.get(section, {}).items():
            for lang in REQUIRED_LANGS:
                if not text.get(lang):
                    errors.append(f"strings.json: {section}.{key} missing {lang!r}")
    for stream in ("wet", "dry", "sanitary", "special_care"):
        if stream not in src.strings.get("streams", {}):
            errors.append(f"strings.json: missing stream name for {stream!r}")
    return errors


def unreviewed(src: KbSource) -> dict[str, list[str]]:
    """Items per language whose text has not been checked by a fluent speaker."""
    out: dict[str, list[str]] = {}
    for item_id, item in src.items.items():
        for lang, ok in item.get("reviewed", {}).items():
            if not ok:
                out.setdefault(lang, []).append(item_id)
    return out


def content_hash(src: KbSource) -> str:
    digest = hashlib.sha256()
    for path in src.files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


SCHEMA_SQL = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE items (
    id TEXT PRIMARY KEY,
    stream TEXT NOT NULL,
    recyclable INTEGER NOT NULL,
    drop_off TEXT,
    json TEXT NOT NULL
);
CREATE TABLE aliases (alias TEXT NOT NULL, lang TEXT NOT NULL, item_id TEXT NOT NULL);
CREATE INDEX aliases_alias ON aliases (alias);
CREATE TABLE detector_classes (class TEXT PRIMARY KEY, item_id TEXT NOT NULL);
CREATE TABLE cities (id TEXT PRIMARY KEY, json TEXT NOT NULL);
"""


def write_sqlite(src: KbSource, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        con.executescript(SCHEMA_SQL)
        con.executemany(
            "INSERT INTO meta VALUES (?, ?)",
            [
                ("kb_version", content_hash(src)),
                ("strings", json.dumps(src.strings, ensure_ascii=False)),
                ("item_count", str(len(src.items))),
            ],
        )
        for item_id, item in sorted(src.items.items()):
            con.execute(
                "INSERT INTO items VALUES (?, ?, ?, ?, ?)",
                (
                    item_id,
                    item["stream"],
                    int(item["recyclable"]),
                    item["drop_off"],
                    json.dumps(item, ensure_ascii=False),
                ),
            )
            terms: set[tuple[str, str]] = set()
            for lang, name in item["names"].items():
                terms.add((name.casefold(), lang))
            for lang, aliases in item.get("aliases", {}).items():
                terms.update((a.casefold(), lang) for a in aliases)
            con.executemany(
                "INSERT INTO aliases VALUES (?, ?, ?)",
                [(alias, lang, item_id) for alias, lang in sorted(terms)],
            )
            con.executemany(
                "INSERT INTO detector_classes VALUES (?, ?)",
                [(cls, item_id) for cls in item["detector_classes"]],
            )
        for city_id, city in sorted(src.cities.items()):
            con.execute(
                "INSERT INTO cities VALUES (?, ?)", (city_id, json.dumps(city, ensure_ascii=False))
            )
        con.commit()
    finally:
        con.close()
    tmp.replace(out)


def build(kb_dir: Path = KB_DIR, out: Path = DEFAULT_OUT) -> KbSource:
    """Validate and write the database. Raises ValueError listing every problem."""
    src = load_source(kb_dir)
    errors = validate(src, kb_dir)
    if errors:
        raise ValueError("Knowledge base is invalid:\n  " + "\n  ".join(errors))
    write_sqlite(src, out)
    return src


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="validate only, write nothing")
    args = parser.parse_args(argv)

    src = load_source()
    errors = validate(src)
    if errors:
        print("Knowledge base is invalid:", *errors, sep="\n  ", file=sys.stderr)
        return 1
    for lang, ids in sorted(unreviewed(src).items()):
        print(f"note: {len(ids)} items have unreviewed {lang!r} text")
    if not args.check:
        write_sqlite(src, args.out)
        print(f"wrote {args.out} ({len(src.items)} items, version {content_hash(src)})")
    else:
        print(f"ok: {len(src.items)} items, {len(src.cities)} cities")
    return 0


if __name__ == "__main__":
    sys.exit(main())
