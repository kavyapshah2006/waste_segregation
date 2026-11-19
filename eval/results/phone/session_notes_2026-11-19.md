# Phone session, 2026-11-19 (realme RMX3933, Unisoc T612, 4 GB RAM, Android 14)

Ended early (time). Measured on the phone with the in-app timing overlay and adb:

- Detector (YOLO26n, 640 px, XNNPACK): 226-420 ms per frame (overlay readings).
- Llama 3.2 1B SpinQuant: load ~30 s; first token 15.7 s; decode 4.4 tokens/s (first scan).
- App memory with Llama loaded: ~1.2-1.3 GB RSS (dumpsys meminfo); device MemAvailable 0.6-1.6 GB.
- Airplane mode: the app worked end to end with airplane mode on (user check; screen recording by user).
- Accuracy on real items: see items_2026-11-19.csv. Bin right on all but the computer mouse
  (dry instead of special care); calculator named "mobile phone" (no calculator item, same bin).
- Not measured: 15-minute sustained run, timing export, battery drain (phone was charging over USB).
- Bugs found and fixed during the session: Android regex flag (Llama failed), ExecuTorch thread-pool
  deadlock while Llama loaded (one-model lock), shutter used a stale frame and crashed on a closed
  image, detector time included waits.
