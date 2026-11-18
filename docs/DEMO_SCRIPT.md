# Demo video script (about 3 minutes)

Record on the phone with screen recording on, airplane mode visibly on from the first shot. One take
per scene; cut between scenes. Show the real app, no mock-ups. The detector knows 14 TACO classes
(PET bottle, newspaper, cardboard box, paper cup, carry bag, chips packet, Tetra Pak, aluminium can,
tin can, glass bottle, broken glass, thermocol, aluminium foil, used tissue); everything else is
shown through the "not sure" suggestions, a photo, or voice.

| # | Time | Shot | What to show | Say / caption |
|---|---|---|---|---|
| 1 | 0:00 | Phone, quick settings | Turn airplane mode on, Wi-Fi off | "Everything runs on this phone. No internet." |
| 2 | 0:10 | Table of mixed waste, **Scan** tab | Point at a PET bottle, then a chips packet | The card appears with its bin colour first; the timing overlay shows detector and card times. "The detector names it; the knowledge base decides the bin." |
| 3 | 0:30 | Same card | Llama's explanation streams in under the steps | "Llama 3.2 explains, on the phone's CPU, through ExecuTorch. Every sentence is checked against the knowledge base." |
| 4 | 0:50 | Greasy cardboard (pizza box piece) | The card asks "Is it dirty?" → tap **Yes** | Card switches bin and steps. "Condition matters, and when it can't tell, it asks." |
| 5 | 1:10 | Banana peel | "Not sure? Suggest" → three suggestions → tap *Fruit and vegetable peels* | "When unsure it asks instead of guessing." |
| 6 | 1:30 | Wrong card (any) | Tap **This is wrong** → pick the right item | "Corrections stay on the phone and can be exported to improve the model." |
| 7 | 1:45 | **Ask** tab, Hindi | Tap **🎤 Speak**, say "मेरे पास पुरानी बैटरी है, इसे कहाँ डालूँ?" | Battery card in Hindi, read aloud, Hindi explanation. "Hindi voice in and out, offline." |
| 8 | 2:15 | **Ask** tab | Type "What is the capital of France?" | Polite decline. "It stays on topic." |
| 9 | 2:30 | **Impact** tab | Scroll | Items by stream, estimated kg kept out of landfill (labelled as an estimate), streak, today's quiz. |
| 10 | 2:50 | **Settings** tab | Models list (all "ready"), "Built with Llama" line | End card: repo link, "ExecuTorch on Arm". |

Before recording: charge to 100%, note the battery level at start and end, close other apps, run
`./scripts/phone_collect.sh start` (records the phone model in `docs/DEVICE.md`).
