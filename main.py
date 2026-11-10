"""Human detection with a YOLOv8 model on Raspberry Pi.

Detects people in a camera feed, estimates how far away each one is, and
lights an LED for each distance zone someone is standing in:
    red   (pin 15 / GPIO22)  1 m > RED
    green (pin 13 / GPIO27)  1 m < GREEN < 2 m
    blue  (pin 11 / GPIO17)  > 2 m BLUE
Several LEDs can be on at once when people are in different zones.
An active buzzer, switched by a transistor on pin 16 (GPIO23), beeps for
BUZZER_DURATION_S seconds when a human appears.

Wiring (physical pin numbers on the 40-pin header):
    pin 11 (GPIO17) -> resistor -> blue LED  -> GND
    pin 13 (GPIO27) -> resistor -> green LED -> GND
    pin 15 (GPIO22) -> resistor -> red LED   -> GND
    pin 16 (GPIO23) -> ~1k resistor -> NPN transistor base
    pin 2  (5V)     -> buzzer (+); buzzer (-) -> transistor collector
    pin 39 (GND)    -> transistor emitter and the LED grounds

How each frame is handled:
    1. Grab a frame from the camera (or webcam / video file).
    2. Find people with the YOLO model and faces with OpenCV's YuNet model.
    3. Estimate each person's distance from the width of their face. Up
       close a whole body doesn't fit in frame, so a body's height is only
       used when the person is fully in view and no face is visible.
    4. Put each person in a zone (red, green, blue) by distance.
    5. Light the matching LEDs and beep if someone has just appeared.
    6. Draw the boxes, distances and FPS on the preview, tinted red if
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
    python main.py --imgsz 416          # bigger YOLO input: farther people, lower FPS
    python main.py --detect-every 1     # run the models on every frame (default 2)
    python main.py --face-scale 1       # full-size face detection (default 0.5)

Speed: the models, not the camera, limit the frame rate. YOLO runs from the
NCNN export in models/human_best_ncnn_model when it exists (about 3x faster
than the .pt on a Pi). NCNN models have a fixed input size, so after changing
--imgsz, re-export to match:
    yolo export model=models/human_best.pt format=ncnn imgsz=416

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
import time
from collections.abc import Mapping, Sequence
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
# Run the models on every Nth frame and reuse the last result in between.
DEFAULT_DETECT_EVERY = 2
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
LED_PINS = {"red": 22, "green": 27, "blue": 17}
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
# It beeps once for BUZZER_DURATION_S seconds when a human appears; someone
# staying in view doesn't keep it beeping.
BUZZER_PIN = 23
BUZZER_DURATION_S = 2.0
# Keep the LEDs on for 3 s after the last detection. This also means a single
# missed frame doesn't make them flicker.
DETECTION_HOLD_S = 3.0

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

    def __init__(self, width: int = 640, height: int = 480, fps: int = 60) -> None:
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


def open_source(source: str, width: int, height: int) -> FrameSource:
    """Open the frame source chosen with --source.

    "picam" uses the Pi Camera, a number is a USB webcam index, and anything
    else is treated as a video or image file path.
    """
    if source == "picam":
        return PiCamera(width, height)
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


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


def draw_hud(frame: np.ndarray, fps: float, count: int) -> None:
    """Draw the FPS and people count in the top-left corner."""
    cv2.putText(frame, f"FPS {fps:.1f}  Humans {count}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, HUD_COLOR_BGR, 2)


def status_line(fps: float, people: Sequence[Person]) -> str:
    """One-line text summary used with --no-show."""
    details = "  ".join(describe(p) for p in people)
    return f"FPS {fps:.1f}  humans={len(people)}  {details}".rstrip()


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
    parser.add_argument("--detect-every", type=int, default=DEFAULT_DETECT_EVERY,
                        help="run the models on every Nth frame "
                             f"(default {DEFAULT_DETECT_EVERY})")
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
    if args.detect_every < 1:
        parser.error("--detect-every must be at least 1")
    if not 0 < args.face_scale <= 1:
        parser.error("--face-scale must be between 0 and 1")
    return args


def run(
    args: argparse.Namespace,
    model: YOLO,
    faces: FaceDetector,
    cap: FrameSource,
    leds: Indicators,
) -> None:
    """Detect, light and display frames until the source ends or the user quits."""
    show = not args.no_show
    fullscreen = args.fullscreen
    if show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        set_fullscreen(WINDOW_NAME, fullscreen)

    alert = PresenceAlert()
    prev_time = time.monotonic()
    people: list[Person] = []
    frame_index = 0
    avg_dt = 0.0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            log.info("Video source ended.")
            return

        height, width = frame.shape[:2]
        # The models are the slow part, so they only run on every Nth frame;
        # frames in between reuse the last result.
        detect_now = frame_index % args.detect_every == 0
        frame_index += 1
        if detect_now:
            face_boxes = faces.detect(frame, args.face_scale)
            people = locate_people(
                detect_humans(model, frame, args.conf, args.imgsz),
                face_boxes,
                height,
                focal_for_width(args.focal_px, width),
            )
        zones = {p.zone for p in people if p.zone}

        now = time.monotonic()
        alert.update(now, bool(people))
        if people:
            leds.show(*zones)
        elif not alert.is_holding(now):
            leds.show()
        leds.buzz(alert.is_buzzing(now))

        # Average the time between frames, not the rate: frames that skip
        # detection, or that queued up during one, arrive almost instantly.
        dt = now - prev_time
        avg_dt = dt if avg_dt == 0.0 else 0.95 * avg_dt + 0.05 * dt
        fps = 1.0 / max(avg_dt, 1e-6)
        prev_time = now

        if args.calibrate and detect_now:
            print(calibration_line(face_boxes, args.calibrate, width), flush=True)
        if not show:
            print(status_line(fps, people), flush=True)
            continue

        if TOO_CLOSE_ZONE in zones:
            tint_too_close(frame)
        draw_people(frame, people)
        draw_hud(frame, fps, len(people))
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
            cap = open_source(args.source, args.width, args.height)
            cleanup.callback(cap.release)
        except (OSError, RuntimeError, cv2.error) as exc:
            log.error("Startup failed: %s", exc)
            return 1
        cleanup.callback(cv2.destroyAllWindows)

        log.info("Model: %s  Classes: %s", args.model, model.names)
        log.info("YOLO size: %d  Face scale: %g  Detect every %d frame(s)",
                 args.imgsz, args.face_scale, args.detect_every)
        log.info("Focal length: %.0f px at %d px wide", args.focal_px, FOCAL_REFERENCE_WIDTH_PX)
        log.info("Press the space bar in the window (or Ctrl+C) to quit, 'f' to toggle full screen.")
        try:
            run(args, model, faces, cap, leds)
        except KeyboardInterrupt:
            log.info("Interrupted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
