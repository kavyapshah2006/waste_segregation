# Llama answer-quality experiments (2026-11-18)

How to get the most useful answers out of the stock Llama 3.2 1B models, without fine-tuning.
Every variant ran on the same **88 on-topic validation questions** (held-out items: broken glass,
chips packet, fruit/veg peels, glass bottle, nail polish, old clothes, pesticide container, carry
bag, tampon; English and Hindi; plain and dirty/broken conditions; explain, which bin, recyclable,
prepare). Off-topic questions never reach Llama (the app declines them first), so they are not
counted here. Harness: [`eval/llm_quality/experiments.py`](../../llm_quality/experiments.py);
each `<model>__<variant>.json` holds the scores and `.answers.jsonl` the raw answers.

**Useful** = valid JSON, passes every verifier check (bin, stream, recyclable/compostable,
negated steps, unsafe advice), non-empty `why`, in the requested language. These are word rules;
see the hand check below. Laptop runs (ExecuTorch 1.5.1, x86); answers on the phone use the same
`.pte` files, so quality carries over, but speed does not.

## Results

| model | variant | useful | useful en | useful hi | valid JSON | tokens/answer | tries/answer |
|---|---|---|---|---|---|---|---|
| SpinQuant | **examples + rules + sampled retry (shipped)** | **96.6%** | 93.2% | 100% | 100% | 66 | 1.09 |
| SpinQuant | examples + sampled retry | 97.7% | 95.5% | 100% | 100% | 67 | 1.12 |
| SpinQuant | 3 sampled tries (T=0.7), no examples | 96.6% | 100% | 93.2% | 100% | 108 | 1.40 |
| QLoRA | examples + sampled retry | 95.5% | 90.9% | 100% | 100% | 71 | 1.16 |
| SpinQuant | examples + rules | 95.5% | 90.9% | 100% | 100% | 63 | 1.00 |
| SpinQuant | examples (same question kind) | 92.0% | 88.6% | 95.5% | 100% | 61 | 1.00 |
| QLoRA | examples (same question kind) | 89.8% | 88.6% | 90.9% | 100% | 62 | 1.00 |
| SpinQuant | greedy, then sampled retries | 88.6% | 90.9% | 86.4% | 98.9% | 110 | 1.48 |
| QLoRA | rules | 87.5% | 88.6% | 86.4% | 96.6% | 74 | 1.00 |
| QLoRA | facts as plain lines | 83.0% | 86.4% | 79.5% | 95.5% | 77 | 1.00 |
| QLoRA | baseline prompt | 81.8% | 88.6% | 75.0% | 95.5% | 77 | 1.00 |
| SpinQuant | rules | 76.1% | 68.2% | 84.1% | 100% | 65 | 1.00 |
| QLoRA | fewer facts | 75.0% | 81.8% | 68.2% | 92.0% | 77 | 1.00 |
| SpinQuant | facts as plain lines | 75.0% | 70.5% | 79.5% | 97.7% | 82 | 1.00 |
| SpinQuant | temperature 0.3 | 72.7% | 70.5% | 75.0% | 95.5% | 72 | 1.00 |
| SpinQuant | 2 fixed examples (not matched to the question) | 71.6% | 50.0% | 93.2% | 95.5% | 79 | 1.00 |
| SpinQuant | fewer facts | 70.5% | 70.5% | 70.5% | 97.7% | 69 | 1.00 |
| SpinQuant | **baseline prompt** | **69.3%** | 65.9% | 72.7% | 96.6% | 76 | 1.00 |

Full table with every column: `uv run python eval/llm_quality/experiments.py --summary`.

## What worked and what did not

- **Worked examples matched to the question kind** are the big win (+23 points on SpinQuant):
  two finished exchanges about newspaper and an AA battery, asking the same kind of question in
  the same language. Fixed examples that don't match the question barely help English (50%).
- **Stricter rules** in the system prompt help on their own for QLoRA (+6) and add a little on
  top of the examples.
- **A sampled retry only when the checks fail** adds about one point for ~9% more generations.
  Sampling from the start (3 tries at T=0.7) reaches the same quality at 1.6x the tokens.
- **Facts as plain lines, fewer facts, temperature 0.3** did not help.
- **A first sentence written from the KB** ("No, it cannot be recycled.") broke the JSON in 8–16
  of 88 answers: after a trailing space the model started `"tip"` without closing `why`. The fix
  (no trailing space) was not re-measured in the 2-day plan; the examples made it unnecessary.
- Differences of 1–2 points are 1–2 questions out of 88: treat the top four rows as a tie.

**Shipped** (pipeline/prompt.py, pipeline/shots.py, pipeline/assistant.py and the Kotlin port):
SpinQuant 1B, examples + rules, greedy first, up to two sampled tries (T=0.7) when the checks
fail. Whatever still fails is never shown: the verifier falls back to the KB's own text.

## Hand check

16 random answers of the shipped setup (8 English, 8 Hindi) read by hand: all consistent with
the KB facts; two Hindi answers have awkward grammar ("सैनिटरी कचरा को…"), one tip on a broken
glass bottle talks about recycling whole bottles. The Hindi text still needs a native speaker.
