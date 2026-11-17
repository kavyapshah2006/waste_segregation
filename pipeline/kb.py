"""Knowledge base lookup and decision resolution (FuturePlan section 5.2-5.3).

Reads the compiled kb.sqlite, the same file the Android app ships with.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_CITY = "default_in"
FALLBACK_LANG = "en"


@dataclass(frozen=True)
class Verdict:
    item_id: str
    name: str
    stream: str
    stream_name: str
    bin_colour: str
    bin_colour_name: str
    recyclable: bool
    compostable: bool
    steps: tuple[str, ...]
    hazard: str | None
    drop_off: str | None
    drop_off_text: str | None
    notes: tuple[str, ...]
    impact: str
    language: str
    city: str
    applied: tuple[str, ...]
    missing_translation: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("steps", "notes", "applied"):
            d[key] = list(d[key])
        return d

    def facts(self) -> dict:
        """The facts handed to Llama: everything it may talk about, nothing else."""
        return {
            "item": self.name,
            "stream": self.stream_name,
            "bin": self.bin_colour_name,
            "recyclable": self.recyclable,
            "steps": list(self.steps),
            "hazard": self.hazard,
            "drop_off": self.drop_off_text,
            "notes": list(self.notes),
            "impact": self.impact,
        }


class _Text:
    """Picks text in the requested language, falling back to English."""

    def __init__(self, lang: str) -> None:
        self.lang = lang
        self.missing = False

    def __call__(self, by_lang: dict | None):
        if by_lang is None:
            return None
        if self.lang in by_lang:
            return by_lang[self.lang]
        self.missing = True
        return by_lang[FALLBACK_LANG]


class KnowledgeBase:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        try:
            self._items = {
                row[0]: json.loads(row[1]) for row in con.execute("SELECT id, json FROM items")
            }
            self._cities = {
                row[0]: json.loads(row[1]) for row in con.execute("SELECT id, json FROM cities")
            }
            self._aliases = con.execute("SELECT alias, lang, item_id FROM aliases").fetchall()
            self._classes = dict(con.execute("SELECT class, item_id FROM detector_classes"))
            meta = dict(con.execute("SELECT key, value FROM meta"))
        finally:
            con.close()
        self.version = meta["kb_version"]
        self.strings = json.loads(meta["strings"])

    @property
    def item_ids(self) -> list[str]:
        return sorted(self._items)

    def item(self, item_id: str) -> dict:
        return self._items[item_id]

    def item_for_class(self, detector_class: str) -> str | None:
        return self._classes.get(detector_class)

    def search(self, text: str, lang: str | None = None) -> list[str]:
        """Item ids whose name or alias matches the text: exact matches first, then partial."""
        query = text.strip().casefold()
        if not query:
            return []
        rows = [r for r in self._aliases if lang is None or r[1] == lang]
        exact = sorted({item_id for alias, _, item_id in rows if alias == query})
        partial = sorted(
            {
                item_id
                for alias, _, item_id in rows
                if item_id not in exact and (query in alias or alias in query)
            }
        )
        return exact + partial

    def resolve(
        self,
        item_id: str,
        attributes: dict | None = None,
        language: str = "en",
        city_id: str = DEFAULT_CITY,
    ) -> Verdict:
        """Decide stream, bin and instructions for an item in a given condition and city."""
        item = self._items[item_id]
        city = self._cities[city_id]
        attributes = attributes or {}
        text = _Text(language)

        stream = item["stream"]
        recyclable = item["recyclable"]
        compostable = item.get("compostable", False)
        drop_off = item["drop_off"]
        hazard = item["hazard"]
        steps_by_lang = item["prep_steps"]
        extra_steps: list[str] = []
        notes: list[str] = []
        applied: list[str] = []

        override = city.get("overrides", {}).get(item_id)
        if override:
            if "stream" in override and override["stream"] != stream:
                stream = override["stream"]
                drop_off = None
            recyclable = override.get("recyclable", recyclable)
            drop_off = override.get("drop_off", drop_off)
            if "note" in override:
                notes.append(text(override["note"]))
            applied.append(f"city:{city_id}")

        for rule in item.get("condition_rules", []):
            condition = rule["if"]
            if not all(attributes.get(key) == value for key, value in condition.items()):
                continue
            then = rule["then"]
            if "stream" in then and then["stream"] != stream:
                stream = then["stream"]
                drop_off = None
            recyclable = then.get("recyclable", recyclable)
            compostable = then.get("compostable", compostable)
            hazard = then.get("hazard", hazard)
            drop_off = then.get("drop_off", drop_off)
            if "steps" in then:
                steps_by_lang = then["steps"]
            if "extra_steps" in then:
                extra_steps.extend(text(then["extra_steps"]))
            if "note" in then:
                notes.append(text(then["note"]))
            applied.append(
                "rule:" + ",".join(f"{k}={str(v).lower()}" for k, v in condition.items())
            )

        colour = city["bin_colours"][stream]
        drop_off_text = text(city["drop_off_info"][drop_off]) if drop_off else None
        return Verdict(
            item_id=item_id,
            name=text(item["names"]),
            stream=stream,
            stream_name=text(self.strings["streams"][stream]),
            bin_colour=colour,
            bin_colour_name=text(self.strings["colours"][colour]),
            recyclable=recyclable,
            compostable=compostable,
            steps=tuple(extra_steps + list(text(steps_by_lang))),
            hazard=text(hazard),
            drop_off=drop_off,
            drop_off_text=drop_off_text,
            notes=tuple(notes),
            impact=text(item["impact"]),
            language=language,
            city=city_id,
            applied=tuple(applied),
            missing_translation=text.missing,
        )

    def language_name(self, language: str) -> str:
        return self.strings["languages"].get(language, language)

    def message(self, key: str, language: str) -> str:
        """A fixed app message (strings.json "messages"), falling back to English."""
        by_lang = self.strings["messages"][key]
        return by_lang.get(language, by_lang[FALLBACK_LANG])

    def names_of(self, item_id: str) -> list[str]:
        """Every name and alias of an item, in every language."""
        item = self._items[item_id]
        names = list(item["names"].values())
        for aliases in item.get("aliases", {}).values():
            names.extend(aliases)
        return names
