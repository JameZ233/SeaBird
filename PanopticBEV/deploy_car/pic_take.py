"""
RC car data logger (minimal)
- Captures frames from a camera (Picamera2 on Raspberry Pi OR USB webcam fallback)
- Saves images + timestamps to an output folder

Usage (Raspberry Pi with Picamera2):
  python rc_data_logger.py --out data/run_001 --fps 20

Usage (USB webcam):
  python rc_data_logger.py --out data/run_001 --fps 20 --backend opencv --device 0

Notes:
- This logger collects time-synced images.
- For BEV training later, labels are still needed (lane mask / free-space / depth / etc.)
  or pseudo-labels from a classical lane detector.
"""

import os
import time
import json
import argparse
from datetime import datetime

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def picamera2_loop(out_dir: str, fps: float, width: int, height: int, seconds: float):
    from picamera2 import Picamera2
    import cv2  # only for imwrite + BGR conversion

    frames_dir = os.path.join(out_dir, "frames")
    ensure_dir(frames_dir)

    meta_path = os.path.join(out_dir, "meta.jsonl")

    cam = Picamera2()
    config = cam.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"}
    )
    cam.configure(config)
    cam.start()

    period = 1.0 / fps
    t0 = time.time()
    i = 0

    with open(meta_path, "a", encoding="utf-8") as f:
        while True:
            t = time.time()
            if seconds > 0 and (t - t0) >= seconds:
                break

            frame = cam.capture_array()  # RGB
            # save as PNG
            fname = f"{i:06d}.png"
            fpath = os.path.join(frames_dir, fname)

            # OpenCV expects BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(fpath, frame_bgr)

            rec = {
                "idx": i,
                "file": f"frames/{fname}",
                "timestamp_unix": t,
                "timestamp_iso": datetime.fromtimestamp(t).isoformat(),
                "width": width,
                "height": height,
                # "steer": None,    # fill later if you log joystick/RC input
                # "throttle": None  # fill later if you log joystick/RC input
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()

            i += 1

            # simple rate control
            dt = time.time() - t
            sleep_t = max(0.0, period - dt)
            time.sleep(sleep_t)

    cam.stop()

def opencv_loop(out_dir: str, fps: float, device: int, width: int, height: int, seconds: float):
    import cv2

    frames_dir = os.path.join(out_dir, "frames")
    ensure_dir(frames_dir)

    meta_path = os.path.join(out_dir, "meta.jsonl")

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {device}")

    # try to set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    period = 1.0 / fps
    t0 = time.time()
    i = 0

    with open(meta_path, "a", encoding="utf-8") as f:
        while True:
            t = time.time()
            if seconds > 0 and (t - t0) >= seconds:
                break

            ok, frame_bgr = cap.read()
            if not ok:
                print("Frame read failed, stopping.")
                break

            fname = f"{i:06d}.png"
            fpath = os.path.join(frames_dir, fname)
            cv2.imwrite(fpath, frame_bgr)

            h, w = frame_bgr.shape[:2]
            rec = {
                "idx": i,
                "file": f"frames/{fname}",
                "timestamp_unix": t,
                "timestamp_iso": datetime.fromtimestamp(t).isoformat(),
                "width": w,
                "height": h,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()

            i += 1
            dt = time.time() - t
            time.sleep(max(0.0, period - dt))

    cap.release()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output folder, e.g. data/run_001")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = run until Ctrl+C")
    ap.add_argument("--backend", choices=["picamera2", "opencv"], default="picamera2")
    ap.add_argument("--device", type=int, default=0, help="OpenCV camera device index")

    args = ap.parse_args()

    out_dir = args.out
    ensure_dir(out_dir)

    # record a run header
    header = {
        "created": now_str(),
        "backend": args.backend,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "seconds": args.seconds,
        "device": args.device,
    }
    with open(os.path.join(out_dir, "run_header.json"), "w", encoding="utf-8") as f:
        json.dump(header, f, indent=2)

    try:
        if args.backend == "picamera2":
            picamera2_loop(out_dir, args.fps, args.width, args.height, args.seconds)
        else:
            opencv_loop(out_dir, args.fps, args.device, args.width, args.height, args.seconds)
    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C).")
    finally:
        print(f"Saved frames under: {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    main()
