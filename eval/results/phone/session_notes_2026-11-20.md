# Final phone check, 2026-11-20 (realme C61, Unisoc T612, 4 GB)

App built from the commits of 2026-11-19 and 2026-11-20, installed with
`SKIP_MODELS=1 ./scripts/install_app.sh`.

## Found and fixed

| What the user saw | Cause | Fix (commit) |
|---|---|---|
| Scanned a carton after a soap bottle, then the bottle again: the card still showed the carton | Suggestion taps queued behind Llama reading its prompt; the leftover picks landed after "Scan again" | Every scan action gets a token, stale work is dropped; the sheet closes on tap; the card waits at most 400 ms for the model (420cfb4) |
| Picking the carton from the suggestions took several taps | Same: no feedback while the pick waited | Same |
| Second scan of the same item still waited for "Why it matters" | The first explanation was stopped by Scan again before it finished; only finished answers were saved | Unfinished explanations are kept and finished while the camera is idle; whole sentences already written are saved as a partial answer (acb9ab3) |
| Typed or spoken questions ("what do I do with an old plastic carton") answered "I can only help with sorting and disposing of waste" | The off-topic guard knew only a short word list and the current item's own names | Any word of any item's names, one-letter slips, everyday verbs (f283cf0) |
| After tapping Speak, nothing visible for ~10 s | Recording ran the full 8 s; the speech model loaded only after; the button said "Listening…" while transcribing | Stops ~1.2 s after the speaker pauses; model loads during recording; level meter and "Got it! Working out what you said…" (64d1234) |
| A second gallery picture did nothing | Full-size decode on the main thread, silent on failure; no "Looking closer" while waiting for Llama | Decoded off the main thread at ≤1600 px with EXIF rotation; message on failure; hint while Llama finishes |

## Battery

Unplugged at 23:39 IST, airplane mode, app in continuous use (scans, explanations, typed and
spoken questions, gallery). Android counted 4 min 0 s on battery (the user plugged back in after
about 5 minutes):

- Battery charge counter: **50 mAh in 4.0 min** (12.5 mAh/min), screen on throughout.
- Scaled: about **190 mAh per 15 min = 4% of 4,700 mAh**, about 16% per hour of non-stop use.
- Android's per-app estimate: SwachhSathi 33.2 mAh (camera 32.1, CPU 1.05), screen 21.9 mAh.
  The CPU share is an estimate from the phone's power profile and is likely low.
- Level 64% -> 63%; temperature 36.1 -> 39.5 °C.

Raw: `battery_start_2026-11-20_1808.txt`, `battery_end_2026-11-20_1816.txt`,
`batterystats_2026-11-20_1816.txt`.

Not measured: a full 15-minute run on battery; a fresh per-stage timing export.
