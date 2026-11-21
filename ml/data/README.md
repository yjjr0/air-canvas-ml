# Recorded clips

Put JSON exports from the web app's **Collect data** tab here, then retrain:

```bash
cd ml
python train_export.py
```

`train_export.py` loads every `*.json` in this folder, mixes the clips with the
synthetic set (real clips are oversampled 3x), holds out whole recording
sessions for validation, and selects the checkpoint on the real-data
validation score. With real data present, `gesture_meta.json` and the page's
"About the model" section report real-session metrics instead of synthetic ones.

Tips for a useful dataset (~80-120 clips per gesture):

- Record in several sessions (different days, rooms, lighting): sessions are
  the unit of the train/validation split, so one long session gives an
  optimistic score.
- Vary distance to the camera, hand angle, and use both hands.
- "Anything else" (idle) matters as much as the gestures: relaxed hands,
  pointing, scratching your face, reaching for the mouse.
- Swipes and circles should be quick: the model sees about two thirds of a
  second at a time.

Each export is plain JSON: 30 frames per clip at 30 Hz, 21 MediaPipe
landmarks x (x, y, z) per frame, plus the camera aspect ratio and a session id.
