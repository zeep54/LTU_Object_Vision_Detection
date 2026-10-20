"""Human detection with a YOLOv8 model on Raspberry Pi.

Detects people in a camera feed, estimates how far away each one is, and
ranks them by distance onto LEDs wired to the Pi's GPIO header:
    red   (pin 11 / GPIO17)  furthest person
    green (pin 13 / GPIO27)  middle person
    blue  (pin 15 / GPIO22)  closest person
With fewer than three people, colors are handed out in that order starting
from red, so one person lights red and two light red and green.

Usage:
    python main.py                      # Pi Camera (via rpicam-vid)
    python main.py --source 0           # USB webcam index
    python main.py --source video.mp4   # video or image file
    python main.py --no-show            # run without a preview window
    python main.py --no-led             # run without touching GPIO
"""

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

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
LED_PINS = {"red": 17, "green": 27, "blue": 22}
# LED colors handed out by distance, furthest person first.
DISTANCE_COLORS = ("red", "green", "blue")
BOX_COLORS_BGR = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}
UNRANKED_BOX_BGR = (160, 160, 160)  # a 4th+ person, who gets no LED
# Keep the LEDs as they are this long after the last detection so a single
# missed frame doesn't make them flicker.
DETECTION_HOLD_S = 0.5


class StatusLeds:
    """Red/green/blue LEDs on the GPIO header, driven through libgpiod."""

    def __init__(self, chip=GPIO_CHIP, pins=LED_PINS):
        import gpiod
        from gpiod.line import Direction, Value

        self._on, self._off = Value.ACTIVE, Value.INACTIVE
        self.pins = pins
        self.request = gpiod.request_lines(
            chip,
            consumer="human-detection",
            config={
                tuple(pins.values()): gpiod.LineSettings(
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

    def release(self):
        self.show()
        self.request.release()


class NoLeds:
    """Stand-in used with --no-led."""

    def show(self, *colors):
        pass

    def release(self):
        pass


class PiCamera:
    """Reads frames from the Raspberry Pi camera by piping rpicam-vid output."""

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
        data = self.proc.stdout.read(self.frame_size)
        if len(data) < self.frame_size:
            return False, None
        yuv = np.frombuffer(data, dtype=np.uint8).reshape(self.height * 3 // 2, self.width)
        return True, cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    def release(self):
        self.proc.terminate()
        self.proc.wait()


def open_source(source, width, height):
    if source == "picam":
        return PiCamera(width, height)
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


def estimate_distance_m(box_height_px):
    if box_height_px <= 0:
        return None
    return (AVG_HUMAN_HEIGHT_M * FOCAL_LENGTH_PX) / box_height_px


def rank_by_distance(boxes):
    """Return [(box, distance_m, color)] sorted furthest first.

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
    for (x1, y1, x2, y2, conf), distance, color in ranked:
        bgr = BOX_COLORS_BGR.get(color, UNRANKED_BOX_BGR)
        label = f"{color or 'human'} {distance:.1f} m ({conf:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 human detection")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="path to .pt weights")
    parser.add_argument("--source", default="picam",
                        help="'picam', a webcam index, or a video/image path")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--no-show", action="store_true", help="don't open a preview window")
    parser.add_argument("--no-led", action="store_true", help="don't drive the GPIO LEDs")
    args = parser.parse_args()

    leds = NoLeds() if args.no_led else StatusLeds()
    leds.show()
    try:
        model = YOLO(args.model)
        cap = open_source(args.source, args.width, args.height)
    except Exception:
        leds.release()
        raise
    print(f"Model: {args.model}  Classes: {model.names}")
    print("Press 'q' in the window (or Ctrl+C) to quit.")

    prev_time = time.time()
    last_seen = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = model(frame, conf=args.conf, imgsz=args.width, verbose=False)[0]
            boxes = [
                (*map(int, box.xyxy[0].tolist()), float(box.conf[0]))
                for box in result.boxes
            ]
            ranked = rank_by_distance(boxes)
            draw_detections(frame, ranked)

            now = time.time()
            if ranked:
                last_seen = now
                leds.show(*(color for _, _, color in ranked if color))
            elif now - last_seen >= DETECTION_HOLD_S:
                leds.show()

            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(frame, f"FPS {fps:.1f}  Humans {len(boxes)}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if args.no_show:
                print(f"FPS {fps:.1f}  humans={len(ranked)}  "
                      + "  ".join(f"{color or '-'}={d:.1f}m" for _, d, color in ranked))
            else:
                cv2.imshow("Human Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        leds.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
