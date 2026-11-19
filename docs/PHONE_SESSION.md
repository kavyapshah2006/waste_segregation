# Phone session runbook (about 90 minutes)

Everything that needs the real phone, in one sitting: install, accuracy on real items, speed,
memory, battery, the airplane-mode proof and the demo video. Commands run in WSL from the repo
root; `adb.exe` is found automatically in the Windows SDK folder.

## Before you start (laptop, 5 min)

- Phone charged to 100%, other apps closed.
- Laptop: the APK is built (`cd android && ./gradlew :app:assembleRelease`) and the model files
  exist (`models/llm/out/spinquant/`, `models/fallback/out/siglip2_image_int8.pte`).
- Items on the table (the detector knows these 14 TACO classes): PET bottle, newspaper,
  cardboard box, paper cup, carry bag, chips packet, Tetra Pak, aluminium can, tin can, glass
  bottle, aluminium foil, thermocol, used tissue. Plus a banana peel and an AA battery: the
  detector was not trained on them, so they show the "not sure → suggestions" path.

## 1. Connect (5 min)

1. Phone: Settings → About → tap *Build number* 7 times; then Developer options → *USB debugging* on.
2. Plug in USB, accept the "Allow USB debugging?" prompt.
3. Check and record the phone:

```bash
./scripts/phone_collect.sh start
```

This writes `docs/DEVICE.md` (model, chip, RAM, Android version) and resets battery stats.

## 2. Install (10 min, mostly copying 1.2 GB)

```bash
./scripts/install_app.sh
```

Open the app, allow the camera. **Settings** must list every model as `ready`; note the Llama
load time shown there.

## 3. Airplane mode and screen recording (2 min)

Turn on airplane mode and turn Wi-Fi off. Start Android's screen recorder. Keep both on until
step 7 ends: this recording is the "0 bytes to the cloud" proof.

## 4. Accuracy on real items (20 min)

For each item, point the camera at it (one item in view, about 30–50 cm away) until the card
appears. Fill in `eval/results/phone/items_<date>.csv` (copy the template
`eval/results/phone_items_TEMPLATE.csv`): item, what the app said, right or wrong, notes.
Do each item twice: once on the table, once in your hand.

Also try:
- **Photo**: tap *Photo* and pick a picture of an item.
- **Not sure**: hold up the banana peel; when *Not sure? Suggest* appears, tap it and pick.
- **This is wrong**: on one wrong card, tap it and choose the right item.
- **Dirty / broken**: a greasy cardboard piece or a dirty milk pouch; answer the question the card asks.

## 5. Llama (15 min)

On a card, wait for the explanation to finish. Then in **Ask**:
- English: "Can I put this in the recycling?"
- Hindi (switch language in Settings): "इसे फेंकने से पहले क्या करूँ?"
- Off-topic: "What is the capital of France?" (must be declined)

While an explanation is being written, run:

```bash
./scripts/phone_collect.sh memory
```

## 6. Sustained use (15 min)

Keep scanning items for 15 minutes (the overlay shows detector and card times). At the end,
**Settings → Export timings for the report**.

## 7. Collect (5 min)

```bash
./scripts/phone_collect.sh finish
uv run python eval/latency/import_phone.py eval/results/phone/
```

Stop the screen recording; copy it off the phone (it is the airplane-mode proof).

## 8. Demo video (20 min)

Follow [DEMO_SCRIPT.md](DEMO_SCRIPT.md), airplane mode on from the first shot.

## Afterwards

If the phone's owner prefers: `adb.exe uninstall org.wastesegregation.app` (removes the app and
its 1.2 GB of models) and turn USB debugging off.

## Final check (15–20 minutes, after the full session)

1. Connect the phone; on the laptop: installing android/app/build/outputs/apk/release/app-release.apk (models stay on the phone).
2. Open SwachhSathi: scanning should be ready within a few seconds ("Point your camera at a waste item").
3. Scan a bottle: the card appears, and "Why it matters" starts about 1.5 s later; Llama loads the
   first time (about half a minute). Tap **Scan again** while it is writing and scan the next item
   at once: the camera should not wait.
4. Scan the same bottle again: the explanation appears instantly (kept on the phone).
5. Tap the white shutter on a battery: a photo is taken and analysed, no crash.
6. Settings → **Prepare all explanations**, leave it charging if there is time (or before the demo video).
7. Settings → **Export timings for the report**, then `./scripts/phone_collect.sh finish`.
