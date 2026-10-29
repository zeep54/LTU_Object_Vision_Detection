"""Human detection with a YOLOv8 model on Raspberry Pi.

Detects people in a camera feed, estimates how far away each one is, and
ranks them by distance onto LEDs wired to the Pi's GPIO header:
    red   (pin 15 / GPIO22)  furthest person
    green (pin 13 / GPIO27)  middle person
    blue  (pin 11 / GPIO17)  closest person
With fewer than three people, colors are handed out in that order starting
from red, so one person lights red and two light red and green.
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
    2. Run the YOLO model on it to find people.
    3. Estimate each person's distance from the height of their box.
    4. Sort them furthest first and give each a color (red, green, blue).
    5. Light the matching LEDs and beep if someone has just appeared.
    6. Draw the boxes, distances and FPS on the preview window.

Usage (from the project folder, with the venv active, or just type RUN):
    python main.py                      # Pi Camera (via rpicam-vid)
    python main.py --source 0           # USB webcam index
    python main.py --source video.mp4   # video or image file
    python main.py --conf 0.4           # lower the confidence threshold
    python main.py --no-show            # run without a preview window
    python main.py --fullscreen         # full-screen preview ('f' toggles)
    python main.py --no-led             # run without touching GPIO (LEDs or buzzer)

Controls in the preview window:
    space  quit
    f      toggle full screen
"""

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Trained YOLOv8 weights with a single class, "Human".
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "models" / "human_best.pt"

# Distance estimate uses the pinhole camera model:
#   distance = (real_height * focal_length_px) / box_height_px
# Calibrate FOCAL_LENGTH_PX by standing a known distance from the camera:
#   FOCAL_LENGTH_PX = (box_height_px * known_distance_m) / AVG_HUMAN_HEIGHT_M
AVG_HUMAN_HEIGHT_M = 1.7
FOCAL_LENGTH_PX = 600.0

# Physical header pin -> BCM GPIO line: 11 -> 17, 13 -> 27, 15 -> 22.
GPIO_CHIP = "/dev/gpiochip0"  # 40-pin header (RP1) on a Pi 5
LED_PINS = {"red": 22, "green": 27, "blue": 17}
# LED colors handed out by distance, furthest person first.
DISTANCE_COLORS = ("red", "green", "blue")
BOX_COLORS_BGR = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}
UNRANKED_BOX_BGR = (160, 160, 160)  # a 4th+ person, who gets no LED
# Active buzzer, switched by a transistor on physical pin 16 -> GPIO23.
# It beeps once for BUZZER_DURATION_S seconds when a human appears; someone
# staying in view doesn't keep it beeping.
BUZZER_PIN = 23
BUZZER_DURATION_S = 0.5
# Keep the LEDs as they are this long after the last detection so a single
# missed frame doesn't make them flicker.
DETECTION_HOLD_S = 0.5


class StatusLeds:
    """Red/green/blue LEDs and the buzzer on the GPIO header, driven through libgpiod.

    All lines are claimed as outputs, starting off, for as long as the
    program runs. Call release() on exit so they are switched off and freed.
    """

    def __init__(self, chip=GPIO_CHIP, pins=LED_PINS, buzzer_pin=BUZZER_PIN):
        # Imported here so --no-led works on machines without libgpiod.
        import gpiod
        from gpiod.line import Direction, Value

        self._on, self._off = Value.ACTIVE, Value.INACTIVE
        self.pins = pins
        self.buzzer_pin = buzzer_pin
        self.request = gpiod.request_lines(
            chip,
            consumer="human-detection",
            config={
                (*pins.values(), buzzer_pin): gpiod.LineSettings(
                    direction=Direction.OUTPUT, output_value=Value.INACTIVE
                )
            },
        )

    def show(self, *colors):
        """Turn on exactly the given colors and turn the rest off."""
        self.request.set_values({
            line: self._on if name in colors else self._off
            for name, line in self.pins.items()
        })

    def buzz(self, on):
        """Switch the buzzer on (True) or off (False)."""
        self.request.set_value(self.buzzer_pin, self._on if on else self._off)

    def release(self):
        """Turn everything off and hand the GPIO lines back to the system."""
        self.show()
        self.buzz(False)
        self.request.release()


class NoLeds:
    """Stand-in used with --no-led: same methods as StatusLeds, but does nothing."""

    def show(self, *colors):
        pass

    def buzz(self, on):
        pass

    def release(self):
        pass


class PiCamera:
    """Reads frames from the Raspberry Pi camera by piping rpicam-vid output.

    rpicam-vid streams raw YUV420 frames to stdout; each read() takes one
    frame's worth of bytes and converts it to a BGR image for OpenCV. It has
    the same read()/release() interface as cv2.VideoCapture, so the main loop
    doesn't care which one it is using.
    """

    def __init__(self, width=640, height=480, fps=30):
        if shutil.which("rpicam-vid") is None:
            raise RuntimeError("rpicam-vid not found; is the Pi camera stack installed?")
        self.width, self.height = width, height
        self.frame_size = width * height * 3 // 2  # YUV420
        self.proc = subprocess.Popen(
            [
                "rpicam-vid", "-t", "0", "-n",
                "--width", str(width), "--height", str(height),
                "--framerate", str(fps),
                "--codec", "yuv420", "-o", "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self.frame_size,
        )

    def read(self):
        """Return (True, frame) for the next frame, or (False, None) if the stream ended."""
        data = self.proc.stdout.read(self.frame_size)
        if len(data) < self.frame_size:
            return False, None
        yuv = np.frombuffer(data, dtype=np.uint8).reshape(self.height * 3 // 2, self.width)
        return True, cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    def release(self):
        """Stop rpicam-vid."""
        self.proc.terminate()
        self.proc.wait()


def open_source(source, width, height):
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


def estimate_distance_m(box_height_px):
    """Estimate how far away a person is, in meters, from their box height in pixels.

    Assumes the whole person is in frame and about AVG_HUMAN_HEIGHT_M tall,
    so a partly visible person will read as further away than they are.
    Returns None for an empty box.
    """
    if box_height_px <= 0:
        return None
    return (AVG_HUMAN_HEIGHT_M * FOCAL_LENGTH_PX) / box_height_px


def rank_by_distance(boxes):
    """Return [(box, distance_m, color)] sorted furthest first.

    Each box is (x1, y1, x2, y2, confidence) in pixels.

    The furthest people get DISTANCE_COLORS in order; anyone past the
    last color gets None.
    """
    people = [(box, estimate_distance_m(box[3] - box[1])) for box in boxes]
    people = [(box, d) for box, d in people if d is not None]
    people.sort(key=lambda p: p[1], reverse=True)
    return [
        (box, d, DISTANCE_COLORS[i] if i < len(DISTANCE_COLORS) else None)
        for i, (box, d) in enumerate(people)
    ]


def draw_detections(frame, ranked):
    """Draw each person's box and a "color distance (confidence)" label on the frame."""
    for (x1, y1, x2, y2, conf), distance, color in ranked:
        bgr = BOX_COLORS_BGR.get(color, UNRANKED_BOX_BGR)
        label = f"{color or 'human'} {distance:.1f} m ({conf:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)


def set_fullscreen(window, on):
    """Switch the preview window between full screen and a normal window."""
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL)


def main():
    """Parse the options, then run the detect / light / display loop until quit."""
    parser = argparse.ArgumentParser(description="YOLOv8 human detection")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="path to .pt weights")
    parser.add_argument("--source", default="picam",
                        help="'picam', a webcam index, or a video/image path")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--no-show", action="store_true", help="don't open a preview window")
    parser.add_argument("--no-led", action="store_true", help="don't drive the GPIO LEDs or buzzer")
    parser.add_argument("--fullscreen", action="store_true",
                        help="start the preview full screen ('f' toggles)")
    args = parser.parse_args()

    # Set up the GPIO first so all LEDs and the buzzer start off, and release
    # it again if the model or camera fails to load.
    leds = NoLeds() if args.no_led else StatusLeds()
    leds.show()
    try:
        model = YOLO(args.model)
        cap = open_source(args.source, args.width, args.height)
    except Exception:
        leds.release()
        raise
    print(f"Model: {args.model}  Classes: {model.names}")
    print("Press the space bar in the window (or Ctrl+C) to quit, 'f' to toggle full screen.")

    window = "Human Detection"
    fullscreen = args.fullscreen
    if not args.no_show:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        set_fullscreen(window, fullscreen)

    prev_time = time.time()  # for the FPS counter
    last_seen = 0.0          # when a human was last detected
    buzz_until = 0.0         # buzzer stays on until this time
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Detect people; each box becomes (x1, y1, x2, y2, confidence).
            result = model(frame, conf=args.conf, imgsz=args.width, verbose=False)[0]
            boxes = [
                (*map(int, box.xyxy[0].tolist()), float(box.conf[0]))
                for box in result.boxes
            ]
            ranked = rank_by_distance(boxes)
            draw_detections(frame, ranked)

            # Update the LEDs and buzzer. A gap of at least DETECTION_HOLD_S
            # with nobody in view counts as a new arrival, which starts a beep.
            now = time.time()
            if ranked:
                if now - last_seen >= DETECTION_HOLD_S:
                    buzz_until = now + BUZZER_DURATION_S  # a human just appeared
                last_seen = now
                leds.show(*(color for _, _, color in ranked if color))
            elif now - last_seen >= DETECTION_HOLD_S:
                leds.show()
            leds.buzz(now < buzz_until)

            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(frame, f"FPS {fps:.1f}  Humans {len(boxes)}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show the frame (or print a status line with --no-show) and
            # handle key presses.
            if args.no_show:
                print(f"FPS {fps:.1f}  humans={len(ranked)}  "
                      + "  ".join(f"{color or '-'}={d:.1f}m" for _, d, color in ranked))
            else:
                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    break
                if key == ord("f"):
                    fullscreen = not fullscreen
                    set_fullscreen(window, fullscreen)
    except KeyboardInterrupt:
        pass
    finally:
        # Always stop the camera and switch the LEDs and buzzer off.
        cap.release()
        leds.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
