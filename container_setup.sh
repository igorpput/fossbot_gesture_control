#!/usr/bin/env bash
set -e
python3 -m pip install --break-system-packages --ignore-installed mediapipe
wget -O ~/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
echo "Setup complete."
