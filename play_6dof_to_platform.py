"""Preview or stream scaled 6-DoF deck motion to the Stewart platform."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from stewart_controller import Stewart_Platform

DEFAULT_CSV = Path(
    r"D:\study\postgraduate\UAV\3D-VRO\data\reastistic_ship\realistic_ship_6dof.csv"
)
REQUIRED_COLUMNS = [
    "Time(s)",
    "DeckX(m)",
    "DeckY(m)",
    "DeckZ(m)",
    "Roll(rad)",
    "Pitch(rad)",
    "Yaw(rad)",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--port", default="COM7")
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--translation-scale", type=float, default=0.002)
    parser.add_argument("--rotation-scale", type=float, default=0.2)
    parser.add_argument("--send", action="store_true", help="send frames to the ESP32")
    return parser.parse_args()


def main():
    args = parse_args()
    if (
        args.start_time < 0
        or args.seconds <= 0
        or args.playback_speed <= 0
        or args.translation_scale < 0
        or args.rotation_scale < 0
    ):
        raise SystemExit("start-time must be non-negative; seconds and playback-speed must be positive; motion scales must be non-negative")

    data = pd.read_csv(args.csv)
    missing = sorted(set(REQUIRED_COLUMNS).difference(data.columns))
    if missing:
        raise SystemExit(f"CSV is missing columns: {', '.join(missing)}")
    values = data[REQUIRED_COLUMNS].to_numpy(dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise SystemExit("CSV must contain at least two rows of finite values")

    times = values[:, 0]
    if np.any(np.diff(times) <= 0):
        raise SystemExit("Time(s) must be strictly increasing")
    interval_start = times[0] + args.start_time
    interval_end = interval_start + args.seconds
    selected = np.flatnonzero(
        (times >= interval_start - 1e-9) & (times <= interval_end + 1e-9)
    )
    if len(selected) < 2:
        raise SystemExit("Selected playback interval contains fewer than two frames; check start-time and seconds")

    # Start at the platform's calibrated neutral pose; convert CSV metres to
    # millimetres and then apply a conservative scale to the measured motion.
    translation = (values[selected, 1:4] - values[selected[0], 1:4]) * 1000.0
    translation *= args.translation_scale
    rotation = (values[selected, 4:7] - values[selected[0], 4:7]) * args.rotation_scale

    platform = Stewart_Platform(66, 87.5, 45, 115, 0.2269, 0.82, 5 * np.pi / 6)
    frame_angles = []
    for trans, rot in zip(translation, rotation):
        angles = platform.calculate(trans, rot) * 180.0 / np.pi
        if not np.isfinite(angles).all():
            raise SystemExit("Inverse kinematics produced a non-finite angle")
        frame_angles.append(angles.copy())
    frame_angles = np.asarray(frame_angles)

    # The ESP32 firmware maps these IK angles to 90 +/- angle by channel.
    signs = np.array([-1, 1, -1, 1, -1, 1])
    servo_targets = 90.0 + frame_angles * signs
    if servo_targets.min() < 60 or servo_targets.max() > 120:
        raise SystemExit(
            "Calculated servo target exceeds the initial 60-120 degree test window: "
            f"{servo_targets.min():.1f} to {servo_targets.max():.1f} deg"
        )

    source_duration = times[selected[-1]] - times[selected[0]]
    print(
        f"Frames: {len(selected)}, source time: {times[selected[0]]:.2f}-"
        f"{times[selected[-1]]:.2f}s, playback duration: "
        f"{source_duration / args.playback_speed:.2f}s"
    )
    print(f"Geometry: base=66 mm, platform=87.5 mm, horn=45 mm, rod=115 mm")
    print(f"Translation scale: {args.translation_scale}; rotation scale: {args.rotation_scale}")
    print("ESP32 servo target range by channel (degrees):")
    print("min:", np.round(servo_targets.min(axis=0), 2))
    print("max:", np.round(servo_targets.max(axis=0), 2))
    print("First serial frame:", "<A," + ",".join(f"{x:.1f}" for x in frame_angles[0]) + ">")

    if not args.send:
        print("Preview only; no serial data was sent. Add --send after checking the setup.")
        return

    import serial

    with serial.Serial(args.port, 38400, timeout=0.1, dsrdtr=False, rtscts=False) as ser:
        ser.setDTR(False)
        ser.setRTS(False)
        time.sleep(2.0)  # ESP32 may reset when the serial port opens.
        start = time.perf_counter()
        for index, (sample_time, angles) in enumerate(zip(times[selected], frame_angles)):
            deadline = start + (sample_time - times[selected[0]]) / args.playback_speed
            delay = deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            payload = "<A," + ",".join(f"{x:.1f}" for x in angles) + ">"
            ser.write(payload.encode("ascii"))
            if index % 10 == 0:
                print(f"Sent {index + 1}/{len(selected)} frames")

    print("Playback complete. The controller holds the last commanded pose.")


if __name__ == "__main__":
    main()
