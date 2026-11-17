# Android

Kotlin code for the phone app, which runs vision and Llama through ExecuTorch on an Arm phone.

## `core/`: the decision pipeline in plain Kotlin

A Kotlin/JVM library with no Android dependencies. The app (WP5) will use it as is.

| File | Port of | What it does |
|---|---|---|
| `Json.kt` | Python `json` | Reads and writes JSON exactly like Python (key order, escaping, strict parsing), because the prompt text and verifier behaviour must match byte for byte |
| `KnowledgeBase.kt` | `pipeline/kb.py` | Item search, detector class lookup, `resolve()` → `Verdict` (city overrides, condition rules, bin colours, language fallback) |
| `Prompt.kt` | `pipeline/prompt.py` | Llama 3.2 prompt with the prefilled `{"why": "` answer start |
| `Verify.kt` | `pipeline/verify.py` | JSON extraction and repair, bin/stream/recyclable/compostable/step/unsafe checks |
| `Assistant.kt` | `pipeline/assistant.py` | Off-topic guard, one short retry, fallback to KB text |

Reading `kb.sqlite` is platform code and not part of `core`: the tests use JDBC, the app will use `android.database.sqlite`. `KbData` holds the loaded rows.

### Tests

```bash
cd android && ./gradlew :core:test
```

The test task first builds `kb/build/kb.sqlite` with `uv run python kb/build_kb.py`, then checks:

- every golden fixture in `tests/fixtures/golden/` (verdict, exact prompt text, verified answer);
- every case in `tests/fixtures/check_cases.json`, the same file the Python tests read.

A change to the Python pipeline that is not mirrored here fails CI (`kotlin-core` job). Needs JDK 17+ and uv. In WSL, if Gradle cannot download its distribution, set `JAVA_TOOL_OPTIONS=-Djava.net.preferIPv4Stack=true` (Java tries IPv6 first and times out there).

## App

Not created yet (WP5: needs the Android SDK, whose licence you accept when installing it). Prerequisites are in [docs/DEV_SETUP.md](../docs/DEV_SETUP.md).
