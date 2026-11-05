# MemeCV

MemeCV mirrors your webcam feed and shows a matching reaction photo after a gesture holds for seven consecutive processed frames.

## macOS setup

Use native **CPython 3.12** on **macOS 13 Ventura or later**. The pinned dependency set resolved and downloaded as wheels for Apple Silicon (`arm64`) and Intel (`x86_64`) using CPython 3.12 and macOS 13 wheel targets. The OpenCV wheels set the macOS 13 minimum for both architectures. Use an arm64 Python on Apple Silicon for native execution.

The program was smoke-tested with Python 3.12.14 on Linux. Use the checklist below to verify webcam and window behavior on your Mac.

With `MemeCV.zip` in Downloads and Python 3.12 installed, run these commands in Terminal.

```bash
cd ~/Downloads
unzip MemeCV.zip
cd MemeCV
python3.12 -c 'import platform; print(platform.python_version(), platform.machine())'
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: -r requirements.txt
python main.py
```

The pinned MediaPipe wheel includes the models for Pose, Hands, and optional Face Mesh. The only OpenCV package needed is the GUI-enabled `opencv-contrib-python`.

```bash
python main.py --face-landmarks  # White face landmark dots
python main.py --camera 1       # Try another webcam index
```

If the camera fails, enable access for Terminal or your launching IDE in **System Settings → Privacy & Security → Camera**, close other apps using the camera, and retry. The program tries AVFoundation first and then OpenCV's default backend.

## Manual checks

Use even lighting and face the camera by yourself. Start with your hands lowered. The Webcam window shows the current **Candidate** and the **Confirmed** label that controls the Reaction window.

| Check | What to do | Expected reaction |
| --- | --- | --- |
| Waiting | Start with your hands lowered. After a gesture, lower them again and stay neutral. | “Waiting for pose” appears at launch and returns after seven unmatched frames. |
| Shrug | Frame your upper body and both hands. Bend both elbows and hold your open hands outward on opposite sides of your torso, with palms angled up or toward the camera. | `IMG_5342.jpg`. Hide either hand to stop the Shrug candidate. |
| Peace sign | Extend and separate your index and middle fingers into a V. Curl your ring and pinky fingers. Repeat with either hand and with the V tilted sideways. | `IMG_5343.jpg` |
| Hands up | Raise both hands overhead, as if holding something up. Keep your head, shoulders, elbows, and both raised hands in frame. Each wrist should be above your forehead and its elbow. Try it with bent elbows. | `IMG_5344.JPG`. Move one wrist out of frame and verify that Hands up stops appearing as the candidate. |
| Beckon | Hold one hand with the palm angled up or toward the camera. Bend your index finger toward your palm and separate it from your loosely curled fingers. Try a fist, thumbs-up, and flat palm afterward. | `IMG_5345.JPG` for the held beckon. Each negative example should clear the Beckon candidate. |
| Stabilization | Switch from one pose to another, then return to neutral. Watch both labels. | Candidate updates each frame. Confirmed switches after seven consecutive identical candidates. |
| Exit | Focus a MemeCV window and press Esc. | Both windows close and the camera is released. |

Hands up occupies the first priority slot, followed by Shrug, Peace sign, and Beckon. The overhead rule estimates the top of the face from visible Pose eye and nose landmarks. The hand rules estimate palm orientation from the plane of the hand and use joint angles and relative distances. Adjustable thresholds are near the top of `main.py`. Reaction assets are loaded once from `assets/new/` relative to the script.

## Dependency references

- [MediaPipe 0.10.21 wheels](https://pypi.org/project/mediapipe/0.10.21/#files)
- [OpenCV 4.11.0.86 wheels](https://pypi.org/project/opencv-contrib-python/4.11.0.86/#files)
- [NumPy 1.26.4 wheels](https://pypi.org/project/numpy/1.26.4/#files)
- [JAX 0.4.38 wheels](https://pypi.org/project/jax/0.4.38/#files)
- [jaxlib 0.4.38 wheels](https://pypi.org/project/jaxlib/0.4.38/#files)
