# Offline speech (WP9)

## Choice

| Part | Choice | Licence | Size |
|---|---|---|---|
| Speech → text | Meta Omnilingual ASR 300M CTC v2, int8, through sherpa-onnx | Apache-2.0 | 366 MB |
| Transcript → item | `pipeline/voice.py` (Kotlin: `Voice.kt`) | ours | – |
| Text → speech | Android's built-in text-to-speech (Hindi offline voice) | system | 0 MB in the app |

Piper Hindi voices were not chosen for the app: "pratham" and "priyamvada" are trained on CC BY-NC-SA 4.0 data (non-commercial), and the licence of "rohan" (IIT Madras IndicTTS data) could not be checked (the licence PDF did not load). "pratham" is used on the laptop only, to make test audio.

## Laptop round trip (synthetic speech)

```bash
bash models/speech/download.sh
uv run --group speech python models/speech/check_hindi.py --label laptop_tts_roundtrip
```

The Piper voice reads every KB item's Hindi name, the same name inside a sentence ("मेरे पास … है, इसे कहाँ डालूँ?") and 5 questions that name no item. Each recogniser transcribes them. Result (2026-11-18, `eval/results/speech_laptop_tts_roundtrip.json`):

| Recogniser | Size | Median CER | Name, exact search | Name, voice match top 1 / top 3 | Sentence, voice match top 1 / top 3 |
|---|---|---|---|---|---|
| **Omnilingual 300M CTC v2 int8** | 366 MB | 15% | 25% | **92.5% / 97.5%** | **87.5% / 97.5%** |
| Dolphin base CTC int8 | 104 MB | 30% | 2.5% | 50% / 60% | 82.5% / 95% |
| Dolphin small CTC int8 | 250 MB | 65% | 10% | 52.5% / 60% | 27.5% / 50% |
| Whisper base int8 (told "Hindi") | 160 MB | 100% | 0% | 55% / 67.5% | 45% / 67.5% |

No recogniser made the matcher confident about an item on the item-free questions.

What went wrong before the matcher: the CTC models guess the language per utterance and often write a Hindi name in Latin letters ("gatte ka dibba") or slip into another script (Telugu, Sinhala, Urdu); Whisper base mostly writes Urdu script. Exact KB search misses all of these.

**Caveats.** The audio is synthetic, from one clean voice, so these numbers are an upper bound: record native speakers (noisy kitchens included) before quoting anything. The synthetic voice is random per run, so numbers move by a few points between runs. Laptop speed says nothing about the phone.

## Versions

sherpa-onnx is pinned to 1.13.8 in the `speech` group: 2.0 removed eSpeak-NG, which the Piper test voice needs. Recognition works in both; the Android app can use a 2.x AAR for recognition alone.
