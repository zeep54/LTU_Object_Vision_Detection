"""Human detection with a YOLOv8 model on Raspberry Pi.

Detects people in a camera feed, estimates how far away each one is, and
lights an LED for each distance zone someone is standing in:
    red   (pin 15 / GPIO22)  1 m > RED
    green (pin 13 / GPIO27)  1 m < GREEN < 2 m
    blue  (pin 11 / GPIO17)  > 2 m BLUE
Several LEDs can be on at once when people are in different zones.
An active buzzer, switched by a transistor on pin 16 (GPIO23), sounds
while the red LED is lit (someone within 1 m).

Wiring (physical pin numbers on the 40-pin header):
    pin 11 (GPIO17) -> resistor -> blue LED  -> GND
    pin 13 (GPIO27) -> resistor -> green LED -> GND
    pin 15 (GPIO22) -> resistor -> red LED   -> GND
    pin 16 (GPIO23) -> ~1k resistor -> NPN transistor base
    pin 2  (5V)     -> buzzer (+); buzzer (-) -> transistor collector
    pin 39 (GND)    -> transistor emitter and the LED grounds

How it runs: three threads, so slow detection never slows the preview
    camera thread     LatestFrameReader: reads frames from the camera as
                      fast as it delivers them (--fps) and keeps only the
                      newest one. Old frames are dropped, never queued.
    detection thread  DetectionWorker: takes the newest frame, finds faces
                      (OpenCV YuNet) and people (YOLO), works out distances
                      and zones, publishes the result, then grabs the next
                      newest frame. Runs as often as the models allow.
    main thread       For every new camera frame: hand it to the detector,
                      take the latest detection result, drive the LEDs and
                      buzzer, and (at most --view-fps times a second) draw
                      the boxes and show the preview.
    The preview therefore runs at camera speed (capped at --view-fps) while
    boxes update at detection speed. The HUD shows all three rates:
    "Cam" (frames from the camera), "View" (frames shown) and "Detect"
    (detections finished per second).
    Video/image files are the exception: each frame waits for its own
    detection (nothing is dropped), so files play at detection speed.

Per detection:
    1. Find people with the YOLO model and faces with OpenCV's YuNet model.
    2. Estimate each person's distance from the width of their face. Up
       close a whole body doesn't fit in frame, so a body's height is only
       used when the person is fully in view and no face is visible.
    3. Put each person in a zone (red, green, blue) by distance.
Per displayed frame:
    4. Light the matching LEDs and beep if someone has just appeared.
    5. Draw the boxes, distances and FPS on the preview, tinted red if
       anyone is too close.

Usage (from the project folder, with the venv active, or just type RUN):
    python main.py                      # Pi Camera (via rpicam-vid)
    python main.py --source 0           # USB webcam index
    python main.py --source video.mp4   # video or image file
    python main.py --conf 0.4           # lower the confidence threshold
    python main.py --no-show            # run without a preview window
    python main.py --fullscreen         # full-screen preview ('f' toggles)
    python main.py --no-led             # run without touching GPIO (LEDs or buzzer)
    python main.py --calibrate 1        # stand 1 m away; prints --focal-px to use
    python main.py --focal-px 470       # use a calibrated focal length
    python main.py --imgsz 416          # bigger YOLO input: farther people, fewer detections/s
    python main.py --face-scale 1       # full-size face detection (default 0.5)
    python main.py --fps 30 --view-fps 30   # 30 fps preview, ~18 detections/s (default 60/60, ~16/s)
    python main.py --view-fps 0         # no limit on preview frames per second (default 60)
    python main.py --detect-fps 10      # cap detections per second (default 0 = as fast as possible)

Speed: YOLO runs from the NCNN export in models/human_best_ncnn_model when
it exists (about 3x faster than the .pt on a Pi). NCNN models have a fixed
input size, so after changing --imgsz, re-export to match:
    yolo export model=models/human_best.pt format=ncnn imgsz=416

Performance notes (measured 2026-11-07, Pi 5, Camera Module 3 / IMX708, 640x480):
    Before the threaded pipeline the loop ran at ~20 fps. Two causes:
      - rpicam-vid was hard-coded to --framerate 30, so a frame took 33 ms.
      - Everything ran in one loop: read, detect (every 2nd frame), show.
        Two frames + one detection = 33 + 33 + ~60 ms = ~100 ms = 20 fps.
    The camera itself does 120 fps here (577 frames in 5 s, none dropped).
    At 640x480 rpicam-vid uses the 1536x864 sensor mode at both 30 and 120
    fps, so the field of view, and therefore --focal-px, didn't change.
    Model timings (one at a time, real camera frames):
      YOLOv8n NCNN imgsz 320  ~42 ms  (38.7 inference, 3.0 pre, 0.7 post)
      YuNet face, scale 0.5   ~12-15 ms
      => one detection ~55-60 ms => ~17 detections/s ceiling on the CPU.
    LAG FIX (most important finding, 2026-11-07, after the user reported
    the preview was "very laggy" even at an average 30 fps):
      The average fps hid the problem. Gaps between shown frames were
      either ~9 ms or ~50-70 ms: frames arrived in PAIRS. The cause was
      rpicam-vid itself, not detection or the CPU. Without --flush its
      stdout releases each frame only when the next one is written. The
      raw pipe alone, with no Python processing, gave 30 fps as a pair
      every ~64 ms. With --flush (now in PiCamera): one frame every 33 ms,
      max gap 36 ms. The original single-loop code had this bug too.
      Measure smoothness with frame-gap percentiles, not average fps.
      Defaults now (--fps 30, --view-fps 30, --detect-fps 10), preview open:
        shown 30.1 fps, gap p50 34 / p95 39 / max 46 ms, 0 gaps > 50 ms,
        10.1 detections/s. With --detect-fps 0: 16.9/s, still smooth.
    2026-11-10: the user reported choppiness again, the threaded version was
      reverted to the single-loop commit and then brought back. Findings:
      - The choppiness was FULL SCREEN, not the pipeline. At 1920x1080 each
        imshow costs 23-58 ms (Qt scales in software through XWayland; this
        pip cv2 has no Qt OpenGL and no Qt wayland plugin): 21.6 fps shown,
        70/224 gaps > 50 ms. Windowed the same code was smooth. Resizing the
        frame with cv2.resize before imshow only reached 26 fps. A real fix
        needs a GPU-scaled display path (e.g. SDL2) - not done.
      - The single-loop commit (no --flush) at --framerate 60: ~37 fps
        headless, detection every 2nd frame on the main thread.
      - Threaded, preview window open, --detect-fps 0, 22 s runs:
          --fps 30 --view-fps 30: shown 30.0, gap p95 38 / max 42, 18 det/s
          --fps 60 --view-fps 60: shown 59.8, gap p95 22 / max 57, 16 det/s
          --fps 60 --view-fps 30: shown 30.0, gap p95 46 / max 53, 17 det/s
        --no-show at 60 fps: 20 det/s. imshow p95 ~6.5 ms windowed.
      - New defaults: --fps 60 --view-fps 60 --detect-fps 0 (user asked
        for 60 fps and real-time detection). 60 costs ~2 det/s vs 30.
      Tried before finding --flush, no effect on the pairing: capping
      detection, nice-ing the detection thread, NCNN num_threads 2-4,
      NCNN openmp_blocktime 0 vs 20.
    Results with the threaded pipeline, BEFORE the --flush fix (preview
    window open, 14 s runs; averages only, so they hide the pairing):
      --fps --view-fps   Cam   View   Detect/s
        30      30        30    30     17-18    <- default: most detections
        60      30        60    30     15-17
        60      60        60    55-59  11-14    <- smoothest usable preview
       120      30       113    30     ~11
       120      60        83    60     ~9      (camera can't even keep up)
       120   no limit     90    90     ~10
      With --no-show: 30 fps -> 18 detect/s, 120 fps -> 15 detect/s.
      Before this change: 20 fps preview, ~10 detections/s.
    Why more camera fps means fewer detections: every frame is converted
    from YUV, passed through Python and (if shown) copied, drawn and sent
    to the desktop (gnome-shell + Xwayland used ~33% CPU between them).
    That's CPU the models need. Detection can never use more than ~17-18
    frames/s anyway, so camera frames beyond that only smooth the preview.
    CPU was 73 C at 2.4 GHz during these runs, i.e. not throttling. If
    detection is ever much slower than this table, check the temperature
    (cat /sys/class/thermal/thermal_zone0/temp) and the cooling first.
    Things tried and rejected (don't retry without a new reason):
      - Camera at 120 fps by default: costs detections (table above).
      - An uncapped preview: ~90 fps shown, detection fell to ~10/s.
      - Throttling the preview with "has 1/view_fps passed since the last
        shown frame?": jitter made every 2nd frame look early and halved
        the rate (30 fps camera -> 15 shown). Fixed with a schedule plus
        half a frame of slack, see run().
      - Changing NCNN's thread count (2, 3) or cv2.setNumThreads(1): no
        gain. NCNN's default of 4 threads was the fastest.
      - Running YOLO and YuNet in parallel threads: slower (63-69 ms per
        pair vs 54-58 ms sequential). Both use all 4 cores and fight.
        cv2.setNumThreads(1 or 2) didn't fix it.
      - Worrying about the GIL: NCNN and OpenCV release it. With YOLO
        looping in a thread the main thread still got ~93% of its ticks,
        so plain threads are fine and multiprocessing isn't needed.
      - A smaller model: human_best.pt is already YOLOv8n (6 MB, nano).
    Next step for much higher detection rates: a Raspberry Pi AI HAT+
    (Hailo) running the models. Only DetectionWorker's detect function
    would need to change. Caveat: this Pi runs Ubuntu, and Hailo's
    packages target Raspberry Pi OS.

Controls in the preview window:
    space  quit
    f      toggle full screen
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from ultralytics import YOLO

log = logging.getLogger("human-detection")

# --- Configuration -----------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
# Trained YOLOv8 weights with a single class, "Human". The NCNN export of the
# same weights is much faster on the Pi's CPU, so it's used when present.
PT_MODEL = PROJECT_DIR / "models" / "human_best.pt"
NCNN_MODEL = PROJECT_DIR / "models" / "human_best_ncnn_model"
DEFAULT_MODEL = NCNN_MODEL if NCNN_MODEL.is_dir() else PT_MODEL
# YOLO input size. Must match the size the NCNN model was exported at.
DEFAULT_IMGSZ = 320
# YuNet runs on the frame shrunk by this factor; boxes are scaled back up.
DEFAULT_FACE_SCALE = 0.5
# Pi camera frame rate. The IMX708 can do 120 in the 1536x864 sensor mode
# rpicam-vid picks for 640x480, but that is NOT the default on purpose:
# every extra frame costs CPU the models need (120 fps -> ~11 detections/s,
# 30 fps -> ~18; table in the module docstring). Detection always takes the
# newest frame, so a faster camera doesn't make detection itself faster.
# 2026-11-10: default raised to 60 at the user's request (smooth 60 fps
# preview, ~16 detections/s vs ~18 at 30; see the docstring).
DEFAULT_CAMERA_FPS = 60
# Most frames per second to draw in the preview. Drawing and showing a frame
# costs CPU that detection needs. Only matters when --fps is higher.
DEFAULT_VIEW_FPS = 60.0
# Most detections per second; 0 = as fast as possible. Uncapped since
# 2026-11-10 (user wants real-time detection): ~16/s with a smooth 60 fps
# preview. Set e.g. 10 to leave CPU headroom for other programs on the Pi.
DEFAULT_DETECT_FPS = 0.0
# OpenCV's YuNet face detector (opencv_zoo, face_detection_yunet_2023mar).
FACE_MODEL = PROJECT_DIR / "models" / "face_detection_yunet_2023mar.onnx"
FACE_SCORE_THRESHOLD = 0.7
# A face with no YOLO body around it needs more confidence to count as a
# person; background blur and patterns can score around 0.75.
LONE_FACE_SCORE_THRESHOLD = 0.85

# Distance uses the pinhole camera model:
#   distance = (real_size * focal_length_px) / size_in_image_px
# The camera's IMX708 sensor has 1.4 um pixels, and for a 640x480 stream
# rpicam-vid scales a 2304x1728 sensor crop down by 3.6x, so each image pixel
# spans 5.04 um. With a 2.25 mm lens: 2.25 mm / 5.04 um = ~446 px.
# The lens spec is approximate; measure the real value with --calibrate.
LENS_FOCAL_MM = 2.25
SENSOR_PIXEL_UM = 1.4 * 2304 / 640
FOCAL_LENGTH_PX = LENS_FOCAL_MM * 1000 / SENSOR_PIXEL_UM  # at 640 px wide
FOCAL_REFERENCE_WIDTH_PX = 640
AVG_FACE_WIDTH_M = 0.15
AVG_HUMAN_HEIGHT_M = 1.7
# A body box this close to the top or bottom edge is cut off, so its height
# says nothing about distance.
FRAME_EDGE_PX = 5

# Distance zones, closest first: (LED color, upper limit in meters, inclusive).
ZONES = (("red", 1.0), ("green", 2.0), ("blue", float("inf")))
TOO_CLOSE_ZONE = "red"

# Physical header pin -> BCM GPIO line: 11 -> 17, 13 -> 27, 15 -> 22.
GPIO_CHIP = "/dev/gpiochip0"
LED_PINS = {"red": 17, "green": 27, "blue": 22}
BOX_COLORS_BGR = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}
UNKNOWN_BOX_BGR = (160, 160, 160)  # a person whose distance can't be measured
HUD_COLOR_BGR = (255, 255, 255)
TOO_CLOSE_TINT_ALPHA = 0.35

# Active buzzer, switched by an NPN transistor on physical pin 16 -> GPIO23.
# The transistor sits between the buzzer (-) and GND, so the pin controls it:
#   GPIO23 HIGH (3.3 V) -> current into the base -> transistor on  -> buzzer sounds
#   GPIO23 LOW  (0 V)   -> no base current       -> transistor off -> buzzer silent
# The buzzer is therefore active-high and must be held LOW whenever it should
# be quiet, including at startup and exit.
# It sounds only while the red LED is lit (someone within 1.0 m, the red
# zone in ZONES), including the DETECTION_HOLD_S hold after that.
BUZZER_PIN = 23
BUZZER_DURATION_S = 0.5
# Keep the LEDs on for 3 s after the last detection. This also means a single
# missed frame doesn't make them flicker.
DETECTION_HOLD_S = 1.5
WINDOW_NAME = "Human Detection"
QUIT_KEY = ord(" ")
FULLSCREEN_KEY = ord("f")


# --- Data types --------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Box:
    """A box in pixels from one of the models, plus its confidence."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def contains_center_of(self, other: Box) -> bool:
        cx, cy = (other.x1 + other.x2) / 2, (other.y1 + other.y2) / 2
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2


@dataclass(frozen=True, slots=True)
class Person:
    """One person: where they are, how far away, and which zone that puts them in.

    distance_m and zone are None when the distance can't be measured
    (no face visible and the body is cut off by the frame).
    """

    box: Box
    distance_m: float | None
    zone: str | None
    measured_by: str | None  # "face" or "body"


class FrameSource(Protocol):
    """Anything with cv2.VideoCapture's read()/release() interface."""

    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def release(self) -> None: ...


class Indicators(Protocol):
    """The LEDs and buzzer, real or stand-in."""

    def show(self, *colors: str) -> None: ...
    def buzz(self, on: bool) -> None: ...
    def release(self) -> None: ...


# --- Hardware ----------------------------------------------------------------

class StatusLeds:
    """Red/green/blue LEDs and the buzzer on the GPIO header, driven through libgpiod.

    All lines are claimed as outputs, starting off, for as long as the
    program runs. Call release() on exit so they are switched off and freed.
    """

    def __init__(
        self,
        chip: str = GPIO_CHIP,
        pins: Mapping[str, int] = LED_PINS,
        buzzer_pin: int = BUZZER_PIN,
    ) -> None:
        # Imported here so --no-led works on machines without libgpiod.
        import gpiod
        from gpiod.line import Direction, Value

        # active_low=False: ACTIVE drives the pin HIGH, INACTIVE drives it LOW.
        self._on, self._off = Value.ACTIVE, Value.INACTIVE
        self._pins = dict(pins)
        self._buzzer_pin = buzzer_pin
        self._buzzing = False
        self._request = gpiod.request_lines(
            chip,
            consumer="human-detection",
            config={
                (*self._pins.values(), buzzer_pin): gpiod.LineSettings(
                    direction=Direction.OUTPUT,
                    active_low=False,
                    output_value=self._off,  # LOW: LEDs off, transistor off, buzzer silent
                )
            },
        )

    def show(self, *colors: str) -> None:
        """Turn on exactly the given colors and turn the rest off."""
        self._request.set_values({
            line: self._on if name in colors else self._off
            for name, line in self._pins.items()
        })

    def buzz(self, on: bool) -> None:
        """Switch the buzzer on (True) or off (False).

        True drives GPIO23 HIGH so the transistor conducts and the buzzer
        sounds; False drives it LOW so the transistor stops conducting and
        the buzzer goes quiet. The pin is only written when the state changes.
        """
        if on == self._buzzing:
            return
        self._request.set_value(self._buzzer_pin, self._on if on else self._off)
        self._buzzing = on

    def release(self) -> None:
        """Turn everything off and hand the GPIO lines back to the system."""
        self.show()
        # Write LOW directly, whatever we think the state is, so the
        # transistor is definitely off before the line is freed.
        self._request.set_value(self._buzzer_pin, self._off)
        self._buzzing = False
        self._request.release()


class NoLeds:
    """Stand-in used with --no-led: same methods as StatusLeds, but does nothing."""

    def show(self, *colors: str) -> None:
        pass

    def buzz(self, on: bool) -> None:
        pass

    def release(self) -> None:
        pass


class PiCamera:
    """Reads frames from the Raspberry Pi camera by piping rpicam-vid output.

    rpicam-vid streams raw YUV420 frames to stdout; each read() takes one
    frame's worth of bytes and converts it to a BGR image for OpenCV. It has
    the same read()/release() interface as cv2.VideoCapture, so the main loop
    doesn't care which one it is using.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = DEFAULT_CAMERA_FPS) -> None:
        if shutil.which("rpicam-vid") is None:
            raise RuntimeError("rpicam-vid not found; is the Pi camera stack installed?")
        self._width, self._height = width, height
        self._frame_size = width * height * 3 // 2  # YUV420
        self._proc = subprocess.Popen(
            [
                "rpicam-vid", "-t", "0", "-n",
                "--width", str(width), "--height", str(height),
                "--framerate", str(fps),
                "--codec", "yuv420", "-o", "-",
                # --flush is essential. Without it rpicam-vid's buffered
                # stdout releases each frame only when the next one is
                # written, so frames arrive in pairs (30 fps -> a pair every
                # 66 ms): a stuttering, laggy preview even with no detection
                # at all. Measured on the raw pipe, 30 fps: gaps p50 64 ms
                # without, p50 33 / max 36 ms with.
                "--flush",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self._frame_size,
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return (True, frame) for the next frame, or (False, None) if the stream ended."""
        assert self._proc.stdout is not None
        data = self._proc.stdout.read(self._frame_size)
        if len(data) < self._frame_size:
            return False, None
        yuv = np.frombuffer(data, dtype=np.uint8).reshape(self._height * 3 // 2, self._width)
        return True, cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    def release(self) -> None:
        """Stop rpicam-vid, killing it if it doesn't exit promptly."""
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()


def open_source(source: str, width: int, height: int, fps: int) -> tuple[FrameSource, bool]:
    """Open the frame source chosen with --source.

    "picam" uses the Pi Camera, a number is a USB webcam index, and anything
    else is treated as a video or image file path. Also returns whether the
    source is live (a camera): live sources may drop frames to stay current,
    files must not.
    """
    if source == "picam":
        return PiCamera(width, height, fps), True
    is_webcam = source.isdigit()
    cap = cv2.VideoCapture(int(source) if is_webcam else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap, is_webcam


class RateMeter:
    """Counts events per second over the last window_s seconds.

    tick() and rate() may be called from different threads, hence the lock.
    In the first second after start the rate reads low because the window
    isn't full yet.
    """

    def __init__(self, window_s: float = 1.0) -> None:
        self._window_s = window_s
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def _trim(self, now: float) -> None:
        while self._times and now - self._times[0] > self._window_s:
            self._times.popleft()

    def tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._times.append(now)
            self._trim(now)

    def rate(self) -> float:
        with self._lock:
            self._trim(time.monotonic())
            return len(self._times) / self._window_s


class LatestFrameReader:
    """Reads a live source on a background thread, keeping only the newest frame.

    Why: if the main loop reads the camera itself, any slow step (a
    detection, a slow imshow) leaves frames waiting in the rpicam-vid pipe,
    so the picture lags and the camera is throttled. This thread drains the
    camera at full speed instead. read() blocks until a frame newer than the
    one it last returned arrives and returns it; frames nobody asked for in
    time are simply overwritten. Same read()/release() interface as the
    sources it wraps.
    """

    def __init__(self, source: FrameSource) -> None:
        self._source = source
        self.rate = RateMeter()  # frames per second arriving from the camera
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._seq = 0            # number of frames received so far
        self._returned_seq = 0   # seq of the frame read() last returned
        self._ended = False
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop:
                ok, frame = self._source.read()
                if not ok or frame is None:
                    break
                self.rate.tick()
                with self._cond:
                    self._frame = frame
                    self._seq += 1
                    self._cond.notify_all()
        finally:
            # Also reached if read() raises: wake read() so the main loop ends
            # instead of waiting forever. The thread's error is still printed.
            with self._cond:
                self._ended = True
                self._cond.notify_all()

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._cond:
            self._cond.wait_for(lambda: self._seq != self._returned_seq or self._ended)
            if self._seq == self._returned_seq:  # ended with nothing new
                return False, None
            self._returned_seq = self._seq
            return True, self._frame

    def release(self) -> None:
        """Stop the thread, then release the source.

        The source is released only after the thread stops (or is given up on
        after 2 s), because releasing a cv2.VideoCapture while another thread
        is inside its read() can crash. If the thread is stuck in a read,
        releasing the source (killing rpicam-vid) is what unblocks it.
        """
        self._stop = True
        self._thread.join(timeout=2)
        self._source.release()
        self._thread.join(timeout=2)


# --- Detection and distance --------------------------------------------------

class FaceDetector:
    """OpenCV's YuNet face detector, resized to match whatever frames it gets."""

    def __init__(self, model: Path = FACE_MODEL, score: float = FACE_SCORE_THRESHOLD) -> None:
        if not model.is_file():
            raise FileNotFoundError(f"Face model not found: {model}")
        self._detector = cv2.FaceDetectorYN.create(str(model), "", (0, 0), score)
        self._size = (0, 0)

    def detect(self, frame: np.ndarray, scale: float = 1.0) -> list[Box]:
        """Find faces, running on the frame shrunk by `scale` for speed.

        Returned boxes are always in the full frame's pixels.
        """
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        size = (frame.shape[1], frame.shape[0])
        if size != self._size:
            self._detector.setInputSize(size)
            self._size = size
        _, faces = self._detector.detect(frame)
        if faces is None:
            return []
        # Each row is x, y, w, h, 10 landmark coordinates, then the score.
        return [
            Box(int(x / scale), int(y / scale), int((x + w) / scale), int((y + h) / scale),
                float(row[-1]))
            for row in faces
            for x, y, w, h in [row[:4]]
        ]


def detect_humans(model: YOLO, frame: np.ndarray, conf: float, imgsz: int) -> list[Box]:
    """Run the YOLO model on one frame and return every person it found."""
    result = model(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
    return [
        Box(*map(int, box.xyxy[0].tolist()), confidence=float(box.conf[0]))
        for box in result.boxes
    ]


def pinhole_distance_m(real_size_m: float, size_px: int, focal_px: float) -> float | None:
    """Distance in meters to an object of known real size; None for an empty box."""
    if size_px <= 0:
        return None
    return real_size_m * focal_px / size_px


def zone_for(distance_m: float | None) -> str | None:
    """The LED color for a distance, from ZONES; None if the distance is unknown."""
    if distance_m is None:
        return None
    return next(color for color, limit in ZONES if distance_m <= limit)


def locate_people(
    humans: Sequence[Box], faces: Sequence[Box], frame_height: int, focal_px: float
) -> list[Person]:
    """Combine body and face boxes into people with distances and zones.

    A face whose center lies inside a body box is that person's face. Faces
    are preferred for distance because they stay fully in frame up close.
    A confident face with no body around it still counts as a person, since
    the YOLO model can miss someone whose face fills the frame.
    """
    unmatched = sorted(faces, key=lambda f: f.width, reverse=True)
    people = []

    for body in humans:
        face = next((f for f in unmatched if body.contains_center_of(f)), None)
        if face is not None:
            unmatched.remove(face)
            distance = pinhole_distance_m(AVG_FACE_WIDTH_M, face.width, focal_px)
            measured_by = "face"
        elif body.y1 > FRAME_EDGE_PX and body.y2 < frame_height - FRAME_EDGE_PX:
            distance = pinhole_distance_m(AVG_HUMAN_HEIGHT_M, body.height, focal_px)
            measured_by = "body"
        else:
            distance, measured_by = None, None
        people.append(Person(body, distance, zone_for(distance), measured_by))

    for face in unmatched:
        if face.confidence < LONE_FACE_SCORE_THRESHOLD:
            continue
        distance = pinhole_distance_m(AVG_FACE_WIDTH_M, face.width, focal_px)
        people.append(Person(face, distance, zone_for(distance), "face"))

    return people


def focal_for_width(base_focal_px: float, frame_width: int) -> float:
    """Scale a focal length measured at FOCAL_REFERENCE_WIDTH_PX to this frame width."""
    return base_focal_px * frame_width / FOCAL_REFERENCE_WIDTH_PX


def calibration_line(faces: Sequence[Box], known_m: float, frame_width: int) -> str:
    """Report the --focal-px value implied by the largest face at a known distance."""
    if not faces:
        return "calibrate: no face in view"
    width_px = max(f.width for f in faces)
    focal = width_px * known_m / AVG_FACE_WIDTH_M
    focal_ref = focal * FOCAL_REFERENCE_WIDTH_PX / frame_width
    return f"calibrate: face {width_px} px wide at {known_m:g} m -> --focal-px {focal_ref:.0f}"


@dataclass(frozen=True, slots=True)
class Detection:
    """Everything one detection pass found in one frame."""

    people: list[Person]
    faces: list[Box]   # raw face boxes, kept for --calibrate
    frame_width: int


def detect_frame(
    model: YOLO, faces: FaceDetector, frame: np.ndarray, args: argparse.Namespace
) -> Detection:
    """Run both models on one frame and turn the boxes into people.

    The two models run one after the other on purpose: running them in
    parallel threads was measured to be slower (see the module docstring).
    This is the function to replace when moving detection to an accelerator.
    """
    height, width = frame.shape[:2]
    face_boxes = faces.detect(frame, args.face_scale)
    people = locate_people(
        detect_humans(model, frame, args.conf, args.imgsz),
        face_boxes,
        height,
        focal_for_width(args.focal_px, width),
    )
    return Detection(people, face_boxes, width)


class DetectionWorker:
    """Runs detection on a background thread, always on the newest frame.

    submit() hands over a frame; if the worker is busy, the frame waits and
    is replaced by any newer submit, so at most one frame is ever waiting.
    latest() returns the most recent result plus a counter that goes up by
    one per finished detection, so the caller can tell when it's new.

    Only this thread touches the models. If detection raises, the error is
    kept and re-raised from latest()/wait_idle() in the main thread, so it
    stops the program instead of silently freezing the boxes.

    The caller must not modify a frame after submitting it (draw on a copy).
    """

    def __init__(
        self,
        detect: Callable[[np.ndarray], Detection],
        max_rate: float = 0.0,
    ) -> None:
        """max_rate: most detections per second (0 = as fast as possible)."""
        self._detect = detect
        self._interval = 1.0 / max_rate if max_rate > 0 else 0.0
        self.rate = RateMeter()  # detections finished per second
        self._cond = threading.Condition()
        self._pending: np.ndarray | None = None
        self._busy = False
        self._latest: Detection | None = None
        self._count = 0
        self._error: BaseException | None = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="detection", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        next_start = 0.0
        while True:
            with self._cond:
                # Rate cap: idle until the next slot. Frames submitted while
                # waiting just replace each other, so the one detected is
                # still the newest.
                self._cond.wait_for(lambda: self._stop,
                                    timeout=max(0.0, next_start - time.monotonic()))
                self._cond.wait_for(lambda: self._pending is not None or self._stop)
                if self._stop:
                    return
                frame, self._pending = self._pending, None
                self._busy = True
            next_start = time.monotonic() + self._interval
            try:
                result = self._detect(frame)
            except BaseException as exc:  # handed to the main thread
                with self._cond:
                    self._error = exc
                    self._busy = False
                    self._cond.notify_all()
                return
            self.rate.tick()
            with self._cond:
                self._latest = result
                self._count += 1
                self._busy = False
                self._cond.notify_all()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("detection thread failed") from self._error

    def submit(self, frame: np.ndarray) -> None:
        with self._cond:
            self._raise_if_failed()
            self._pending = frame
            self._cond.notify_all()

    def latest(self) -> tuple[int, Detection | None]:
        with self._cond:
            self._raise_if_failed()
            return self._count, self._latest

    def wait_idle(self) -> None:
        """Block until every submitted frame has been processed (used for files)."""
        with self._cond:
            self._cond.wait_for(
                lambda: (not self._busy and self._pending is None) or self._error is not None)
            self._raise_if_failed()

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        # A detection in progress finishes first (~60 ms on the Pi's CPU).
        self._thread.join(timeout=5)


class PresenceAlert:
    """Decides when the LEDs should clear and when the buzzer should sound.

    A gap of at least hold_s with nobody in view counts as a new arrival,
    which starts a beep lasting buzz_s. Kept free of hardware so it is easy
    to test.
    """

    def __init__(self, hold_s: float = DETECTION_HOLD_S, buzz_s: float = BUZZER_DURATION_S) -> None:
        self._hold_s = hold_s
        self._buzz_s = buzz_s
        self._last_seen = float("-inf")
        self._buzz_until = float("-inf")

    def update(self, now: float, people_present: bool) -> None:
        if people_present:
            if not self.is_holding(now):
                self._buzz_until = now + self._buzz_s  # a human just appeared
            self._last_seen = now

    def is_holding(self, now: float) -> bool:
        """True while within hold_s of the last detection."""
        return now - self._last_seen < self._hold_s

    def is_buzzing(self, now: float) -> bool:
        return now < self._buzz_until


# --- Display -----------------------------------------------------------------

def describe(person: Person) -> str:
    """Short label like 'green 1.4 m (face)'."""
    if person.distance_m is None:
        return "human ? m"
    return f"{person.zone} {person.distance_m:.2f} m ({person.measured_by})"


def draw_people(frame: np.ndarray, people: Sequence[Person]) -> None:
    """Draw each person's box and label in their zone's color."""
    for person in people:
        b = person.box
        bgr = BOX_COLORS_BGR.get(person.zone, UNKNOWN_BOX_BGR)
        cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), bgr, 2)
        # Above the box, or inside it when that would collide with the HUD.
        label_y = b.y1 - 8 if b.y1 >= 45 else b.y1 + 55
        cv2.putText(frame, describe(person), (b.x1 + 4, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)


def tint_too_close(frame: np.ndarray) -> None:
    """Wash the whole frame red and add a warning banner, in place."""
    red = np.empty_like(frame)
    red[:] = BOX_COLORS_BGR[TOO_CLOSE_ZONE]
    cv2.addWeighted(red, TOO_CLOSE_TINT_ALPHA, frame, 1 - TOO_CLOSE_TINT_ALPHA, 0, dst=frame)
    cv2.putText(frame, "TOO CLOSE", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, HUD_COLOR_BGR, 3)


@dataclass(frozen=True, slots=True)
class Rates:
    """Frames per second at each stage of the pipeline.

    cam:    frames arriving from the source
    view:   frames the main loop handled (shown, or LEDs updated with --no-show)
    detect: detections finished; how often the boxes/distances update
    """

    cam: float
    view: float
    detect: float

    def __str__(self) -> str:
        return f"Cam {self.cam:.0f}  View {self.view:.0f}  Detect {self.detect:.0f}"


def draw_hud(frame: np.ndarray, rates: Rates, count: int) -> None:
    """Draw the frame rates and people count in the top-left corner."""
    cv2.putText(frame, f"{rates}  Humans {count}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, HUD_COLOR_BGR, 2)


def status_line(rates: Rates, people: Sequence[Person]) -> str:
    """One-line text summary used with --no-show."""
    details = "  ".join(describe(p) for p in people)
    return f"{rates}  humans={len(people)}  {details}".rstrip()


def set_fullscreen(window: str, on: bool) -> None:
    """Switch the preview window between full screen and a normal window."""
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL)


# --- Main loop ---------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 human detection")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="path to .pt weights")
    parser.add_argument("--source", default="picam",
                        help="'picam', a webcam index, or a video/image path")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--width", type=int, default=640, help="frame width in pixels")
    parser.add_argument("--height", type=int, default=480, help="frame height in pixels")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help=f"YOLO input size; smaller is faster (default {DEFAULT_IMGSZ})")
    parser.add_argument("--face-scale", type=float, default=DEFAULT_FACE_SCALE,
                        help="shrink frames by this factor for face detection "
                             f"(default {DEFAULT_FACE_SCALE})")
    # --detect-every was removed with the threaded pipeline: detection now
    # runs as often as the models allow, on the newest frame.
    parser.add_argument("--fps", type=int, default=DEFAULT_CAMERA_FPS,
                        help=f"Pi camera frame rate (default {DEFAULT_CAMERA_FPS})")
    parser.add_argument("--detect-fps", type=float, default=DEFAULT_DETECT_FPS,
                        help="most detections per second; 0 = as fast as possible "
                             f"(default {DEFAULT_DETECT_FPS:g})")
    parser.add_argument("--view-fps", type=float, default=DEFAULT_VIEW_FPS,
                        help="most frames per second to draw in the preview; 0 = no limit "
                             f"(default {DEFAULT_VIEW_FPS:g})")
    parser.add_argument("--no-show", action="store_true", help="don't open a preview window")
    parser.add_argument("--no-led", action="store_true", help="don't drive the GPIO LEDs or buzzer")
    parser.add_argument("--fullscreen", action="store_true",
                        help="start the preview full screen ('f' toggles)")
    parser.add_argument("--focal-px", type=float, default=FOCAL_LENGTH_PX,
                        help=f"camera focal length in pixels at {FOCAL_REFERENCE_WIDTH_PX} px wide "
                             f"(default {FOCAL_LENGTH_PX:.0f})")
    parser.add_argument("--calibrate", type=float, metavar="METERS",
                        help="stand this many meters from the camera; prints the --focal-px to use")
    args = parser.parse_args(argv)
    if args.fps < 1:
        parser.error("--fps must be at least 1")
    if args.detect_fps < 0:
        parser.error("--detect-fps must be 0 (no limit) or more")
    if args.view_fps < 0:
        parser.error("--view-fps must be 0 (no limit) or more")
    if not 0 < args.face_scale <= 1:
        parser.error("--face-scale must be between 0 and 1")
    return args


def run(
    args: argparse.Namespace,
    cap: FrameSource,
    live: bool,
    detector: DetectionWorker,
    leds: Indicators,
) -> None:
    """Show frames, drive the LEDs and feed the detector until the source ends or the user quits.

    cap is a LatestFrameReader for live sources (always the newest frame,
    never blocks on detection) or the plain source for files, where each
    frame waits for its own detection so none are skipped.
    """
    show = not args.no_show
    fullscreen = args.fullscreen
    if show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        set_fullscreen(WINDOW_NAME, fullscreen)

    alert = PresenceAlert()
    lit_zones: set[str] = set()  # LED colors currently on
    loop_rate = RateMeter()  # every frame the loop handles
    view_rate = RateMeter()  # frames actually shown (all of them with --no-show)
    # For files there's no reader thread; the loop's own rate is the source rate.
    cam_rate = cap.rate if isinstance(cap, LatestFrameReader) else loop_rate
    # Showing a frame (copy + draw + imshow) costs CPU that detection needs.
    # Measured: an uncapped ~90 fps preview cut detection from 15/s to 9-10/s.
    # So frames beyond --view-fps still update the LEDs and feed the
    # detector, but aren't drawn.
    # Frames are shown on a fixed schedule (next_show advances by exactly
    # show_interval) with half a camera frame of slack. A plain "has
    # show_interval passed?" check was tried first and halved the rate:
    # with camera fps == view fps, jitter made every 2nd frame look early.
    show_interval = 1.0 / args.view_fps if args.view_fps > 0 else 0.0
    slack = 0.5 / args.fps if live else 0.0
    next_show = float("-inf")
    seen_count = 0  # DetectionWorker count of the last result we acted on
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            log.info("Video source ended.")
            return
        loop_rate.tick()

        # The detector keeps this frame, so from here on draw only on a copy.
        detector.submit(frame)
        if not live:
            detector.wait_idle()
        count, detection = detector.latest()
        new_result = count != seen_count
        seen_count = count
        people = detection.people if detection else []
        zones = {p.zone for p in people if p.zone}

        now = time.monotonic()
        alert.update(now, bool(people))
        if people:
            lit_zones = zones
        elif not alert.is_holding(now):
            lit_zones = set()
        leds.show(*lit_zones)
        # The buzzer mirrors the red LED: it sounds only while someone is
        # in the red zone (1.0 m or closer).
        leds.buzz(TOO_CLOSE_ZONE in lit_zones)

        rates = Rates(cam_rate.rate(), view_rate.rate(), detector.rate.rate())
        if args.calibrate and new_result and detection:
            print(calibration_line(detection.faces, args.calibrate, detection.frame_width),
                  flush=True)
        if not show:
            view_rate.tick()
            # Once per detection, not per frame: at high camera fps the terminal
            # would be flooded with identical lines.
            if new_result:
                print(status_line(rates, people), flush=True)
            continue

        if now < next_show - slack:
            continue
        next_show += show_interval
        if next_show < now:  # fell behind (start-up, slow imshow): restart, don't burst
            next_show = now + show_interval
        view_rate.tick()
        frame = frame.copy()
        if TOO_CLOSE_ZONE in zones:
            tint_too_close(frame)
        draw_people(frame, people)
        draw_hud(frame, rates, len(people))
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == QUIT_KEY:
            return
        if key == FULLSCREEN_KEY:
            fullscreen = not fullscreen
            set_fullscreen(WINDOW_NAME, fullscreen)


def main(argv: Sequence[str] | None = None) -> int:
    """Set everything up, run the loop, and always clean up. Returns an exit code."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    # ExitStack releases whatever was opened, in reverse order, however we
    # leave this block: normal exit, Ctrl+C, or an error during setup.
    with ExitStack() as cleanup:
        try:
            # GPIO first so all LEDs and the buzzer start off.
            leds: Indicators = NoLeds() if args.no_led else StatusLeds()
            cleanup.callback(leds.release)
            leds.show()
            leds.buzz(False)

            model = YOLO(str(args.model), task="detect")
            faces = FaceDetector()
            source, live = open_source(args.source, args.width, args.height, args.fps)
            # Live sources get a reader thread that keeps only the newest frame.
            cap: FrameSource = LatestFrameReader(source) if live else source
            cleanup.callback(cap.release)
        except (OSError, RuntimeError, cv2.error) as exc:
            log.error("Startup failed: %s", exc)
            return 1
        # Callbacks run last-registered-first: the detector stops before the
        # camera is released, and the LEDs are switched off last.
        detector = DetectionWorker(
            lambda frame: detect_frame(model, faces, frame, args),
            # Files wait for every detection, so capping them only slows them down.
            max_rate=args.detect_fps if live else 0.0,
        )
        cleanup.callback(detector.stop)
        cleanup.callback(cv2.destroyAllWindows)

        log.info("Model: %s  Classes: %s", args.model, model.names)
        log.info("YOLO size: %d  Face scale: %g  Camera fps: %d",
                 args.imgsz, args.face_scale, args.fps)
        log.info("Focal length: %.0f px at %d px wide", args.focal_px, FOCAL_REFERENCE_WIDTH_PX)
        log.info("Press the space bar in the window (or Ctrl+C) to quit, 'f' to toggle full screen.")
        try:
            run(args, cap, live, detector, leds)
        except KeyboardInterrupt:
            log.info("Interrupted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
