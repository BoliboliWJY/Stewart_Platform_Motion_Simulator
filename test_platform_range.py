"""Preview or explore platform tilt limits, one direction at a time.

Uses the existing Stewart_py_v1 firmware. No serial commands without --send.
Reported physical limits are operator observations at commanded poses, not
measured platform angles. A single-axis test does not validate combined motion.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from datetime import datetime

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from stewart_controller import Stewart_Platform

AXES = {"roll": 0, "pitch": 1, "yaw": 2}
SIGNS = np.array([-1, 1, -1, 1, -1, 1])


def make_platform():
    return Stewart_Platform(66, 87.5, 45, 115, 0.2269, 0.82, 5 * np.pi / 6)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=["roll", "pitch", "yaw", "both"], default="both")
    parser.add_argument("--port", default="COM7")
    parser.add_argument("--max-angle", type=float, default=15.0, help="platform degrees")
    parser.add_argument("--step", type=float, default=0.5, help="platform degrees per observed step")
    parser.add_argument("--speed", type=float, default=1.0, help="platform degrees per second")
    parser.add_argument("--servo-min", type=float, default=60.0)
    parser.add_argument("--servo-max", type=float, default=120.0)
    parser.add_argument("--auto", action="store_true", help="offline automatic-path preview only; physical boundary runs disabled")
    parser.add_argument("--boundary-margin", type=float, default=2.0, help="platform degrees kept inside model boundary in auto mode")
    parser.add_argument("--servo-speed", type=float, default=10.0, help="maximum target servo degrees/second in auto mode")
    parser.add_argument("--diagnose", action="store_true", help="read COM port feedback only; send no motion commands")
    parser.add_argument("--send", action="store_true")
    args = parser.parse_args(argv)
    if args.auto and args.send:
        parser.error(
            "automatic physical boundary testing is disabled after a mechanical failure. "
            "Finite IK and 0-180 command bounds do not validate servo travel, joint "
            "clearance or load. No serial port opened; no motion commands sent."
        )
    if args.diagnose and args.send:
        parser.error("use --diagnose alone; it sends no motion commands")
    numeric = [args.max_angle, args.step, args.speed, args.servo_min, args.servo_max,
               args.boundary_margin, args.servo_speed]
    if not np.isfinite(numeric).all():
        parser.error("all numeric options must be finite")
    if min(args.max_angle, args.step, args.speed) <= 0 or args.step > args.max_angle:
        parser.error("max-angle, step and speed must be positive; step <= max-angle")
    if not 0 <= args.servo_min < args.servo_max <= 180:
        parser.error("servo limits must satisfy 0 <= min < max <= 180")
    if args.boundary_margin <= 0 or args.servo_speed <= 0:
        parser.error("boundary-margin and servo-speed must be positive")
    if args.max_angle > 90:
        parser.error("max-angle must be <= 90 platform degrees")
    return args


def pose_angles(platform, pose_deg, args):
    with np.errstate(invalid="ignore", divide="ignore"):
        angles = np.rad2deg(platform.calculate(np.zeros(3), np.deg2rad(pose_deg)))
    if not np.isfinite(angles).all():
        raise ValueError("inverse kinematics has no finite solution")
    # Validate the actual one-decimal command, including its rounding.
    angles = np.array([float(f"{x:.1f}") for x in angles])
    targets = (90.0 + angles * SIGNS)[::-1]  # PCA9685 channels 0, 1, ..., 5
    if targets.min() < args.servo_min or targets.max() > args.servo_max:
        raise ValueError(
            f"servo target {targets.min():.1f}..{targets.max():.1f} deg exceeds "
            f"{args.servo_min:g}..{args.servo_max:g} deg"
        )
    return angles, targets


def plan_move(platform, start, end, args):
    distance = float(np.max(np.abs(end - start)))
    # Check the entire move before sending its first frame, at >= 20 Hz and
    # no more than 0.05 platform degrees between checked poses.
    duration = distance / args.speed
    count = max(1, int(np.ceil(duration * 20)), int(np.ceil(distance / 0.05)))
    frames = []
    for index in range(1, count + 1):
        pose = start + (end - start) * index / count
        angles, _ = pose_angles(platform, pose, args)
        frames.append((duration * index / count, angles))
    return frames


def candidates(args):
    count = int(np.ceil(args.max_angle / args.step))
    return [min(index * args.step, args.max_angle) for index in range(1, count + 1)]


def preview(platform, args):
    axes = ["roll", "pitch"] if args.axis == "both" else [args.axis]
    results = []
    pose_angles(platform, np.zeros(3), args)
    for axis in axes:
        for direction in (1, -1):
            current = np.zeros(3)
            passed = 0.0
            reason = "requested max-angle reached; not a mechanical limit"
            for magnitude in candidates(args):
                target = np.zeros(3)
                target[AXES[axis]] = direction * magnitude
                try:
                    plan_move(platform, current, target, args)
                except ValueError as error:
                    reason = str(error)
                    break
                current = target
                passed = direction * magnitude
            result = {"axis": axis, "direction": direction,
                      "last_model_pass_deg": passed, "stop_reason": reason}
            results.append(result)
            print(f"{axis:5} {direction:+d}: last model-pass {passed:+.2f} deg; {reason}")
    return results


def read_angles(ser, timeout=5.0, expected=None, check_stop=None):
    deadline = time.monotonic() + timeout
    byte_count = 0
    samples = []
    last_angles = None
    pending = bytearray()
    while time.monotonic() < deadline:
        if check_stop is not None:
            check_stop()
        chunk = ser.readline()
        byte_count += len(chunk)
        pending.extend(chunk)
        # Serial.readline() may return a partial line on timeout. Keep it until
        # the newline arrives rather than discarding part of the feedback.
        if b"\n" not in pending:
            if len(pending) > 4096:
                if len(samples) < 3:
                    samples.append(repr(bytes(pending[:100])))
                pending.clear()
            continue
        raw, _, remainder = pending.partition(b"\n")
        pending = bytearray(remainder)
        line = raw.decode("ascii", errors="replace").strip()
        if line and len(samples) < 3:
            samples.append(repr(line[:120]))
        if not line.startswith("Angles:"):
            continue
        try:
            values = np.array([float(x.strip()) for x in line[7:].split(",") if x.strip()])
        except ValueError:
            continue
        if values.shape != (6,) or not np.isfinite(values).all():
            continue
        last_angles = values
        if expected is None or np.allclose(values, expected, atol=0.06, rtol=0):
            return values
    if byte_count == 0:
        raise RuntimeError(
            f"COM port opened, but received 0 bytes in {timeout:g}s at 38400 baud. "
            "Check ESP32 power/port and uploaded firmware. Stewart_py_v1 must call "
            "Serial.begin(38400) and showAngles(). No further commands sent."
        )
    if last_angles is not None:
        raise RuntimeError(
            f"ESP32 replied with angles {last_angles.tolist()}, but expected "
            f"{np.asarray(expected).tolist()}. No further commands sent."
        )
    if pending and len(samples) < 3:
        samples.append(repr(bytes(pending[:120])))
    raise RuntimeError(
        f"Received {byte_count} serial bytes, but no valid 'Angles:' line with six numbers. "
        f"Check firmware/38400 baud. Samples: {'; '.join(samples)}. No further commands sent."
    )


def transmit_move(ser, frames, check_stop=None):
    start = time.monotonic()
    for offset, angles in frames:
        if check_stop is not None:
            check_stop()
        delay = start + offset - time.monotonic()
        if delay > 0:
            if check_stop is None:
                time.sleep(delay)
            else:
                wait_until(start + offset, check_stop)
        # The firmware prints continuously; discard old feedback so the final
        # acknowledgement describes this move rather than earlier commands.
        ser.reset_input_buffer()
        payload = "<A," + ",".join(f"{x:.1f}" for x in angles) + ">"
        encoded = payload.encode("ascii")
        if ser.write(encoded) != len(encoded):
            raise RuntimeError("Incomplete serial write; stopped sending.")
    ser.flush()
    if check_stop is None:
        read_angles(ser, expected=frames[-1][1])
    else:
        read_angles(ser, expected=frames[-1][1], check_stop=check_stop)


def wait_until(deadline, check_stop):
    while True:
        check_stop()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.05))


def console_stop():
    # Windows console input can be polled without waiting for Enter. Ctrl+C
    # also raises KeyboardInterrupt through Python's normal signal handler.
    if sys.platform == "win32":
        import msvcrt
        while msvcrt.kbhit():
            if msvcrt.getwch().lower() in ("q", "\x03", "\x1b"):
                raise KeyboardInterrupt


def automatic_plans(platform, args, model_results):
    plans = []
    home = np.zeros(3)
    home_angles, _ = pose_angles(platform, home, args)
    for result in model_results:
        # Stay inside the first contiguous feasible branch reached from home.
        magnitude = max(0.0, abs(result["last_model_pass_deg"]) - args.boundary_margin)
        target = np.zeros(3)
        target[AXES[result["axis"]]] = result["direction"] * magnitude
        legs = []
        for start, end in ((home, target), (target, home)):
            frames = plan_move(platform, start, end, args)
            previous, _ = pose_angles(platform, start, args)
            # Stretch timing to cap per-servo target speed near IK boundaries.
            # Reject branch jumps even when both endpoints have finite IK.
            offset = 0.0
            previous_nominal = 0.0
            limited = []
            for nominal, angles in frames:
                servo_delta = float(np.max(np.abs(angles - previous)))
                if servo_delta > 2.0:
                    raise ValueError("IK branch discontinuity detected; automatic path rejected")
                offset += max(nominal - previous_nominal, servo_delta / args.servo_speed)
                limited.append((offset, angles))
                previous, previous_nominal = angles, nominal
            legs.append(limited)
        assert np.allclose(legs[-1][-1][1], home_angles)
        plans.append({"axis": result["axis"], "target_deg": float(target[AXES[result["axis"]]]),
                      "outbound": legs[0], "return": legs[1]})
        print(f"AUTO {result['axis']} {result['direction']:+d}: target {target[AXES[result['axis']]]:+.2f} deg "
              f"(model margin {args.boundary_margin:g} deg), return to home")
    return plans


def run_automatic(ser, plans, log, save, check_stop=console_stop):
    print("AUTO starts in 3 seconds; q / Esc / Ctrl+C = hold and stop. Watch clearance continuously.")
    wait_until(time.monotonic() + 3.0, check_stop)
    for plan in plans:
        result = {"axis": plan["axis"], "target_deg": plan["target_deg"],
                  "operator_clearance_confirmed": False, "phase": "outbound",
                  "returned_home": False}
        log["directions"].append(result)
        save()
        print(f"AUTO {plan['axis']}: ramping to {plan['target_deg']:+.2f} deg")
        transmit_move(ser, plan["outbound"], check_stop=check_stop)
        result["last_completed_command_deg"] = plan["target_deg"]
        result["phase"] = "holding_edge"
        save()
        wait_until(time.monotonic() + 1.0, check_stop)
        result["phase"] = "returning_home"
        save()
        transmit_move(ser, plan["return"], check_stop=check_stop)
        result["returned_home"] = True
        result["phase"] = "completed"
        save()


def explore_direction(ser, platform, args, axis, direction, log, save, ask=input):
    current = np.zeros(3)
    confirmed = 0.0
    result = {"axis": axis, "direction": direction, "last_confirmed_command_deg": 0.0,
              "stop_reason": "in_progress", "observations": []}
    log["directions"].append(result)
    save()
    print(f"\n{axis} {direction:+d}: Enter = confirm clearance and advance; b = retreat/finish; q = hold/quit.")
    print("At home. Check clearance. If touching/jammed, cut servo power; do not request a return move.")
    for magnitude in candidates(args):
        target = np.zeros(3)
        target[AXES[axis]] = direction * magnitude
        try:
            frames = plan_move(platform, current, target, args)
        except ValueError as error:
            result["stop_reason"] = str(error)
            break
        while True:
            key = ask(f"Current {axis}={current[AXES[axis]]:+.2f}; next={direction*magnitude:+.2f} deg [Enter/b/q]: ").strip().lower()
            if key in ("", "b", "q"):
                break
        if key == "q":
            result["stop_reason"] = "operator_quit_hold"
            save()
            raise KeyboardInterrupt
        if key == "b":
            result["stop_reason"] = "operator_retreat_from_unconfirmed_pose"
            retreat = np.zeros(3)
            retreat[AXES[axis]] = confirmed
            transmit_move(ser, plan_move(platform, current, retreat, args))
            current = retreat
            break
        confirmed = float(current[AXES[axis]])
        result["last_confirmed_command_deg"] = confirmed
        # Record the proposed destination before moving. If interrupted, its
        # partial/unknown state is never reported as a confirmed clear pose.
        result["pending_command_deg"] = direction * magnitude
        save()
        transmit_move(ser, frames)
        current = target
        result["observations"].append({"command_deg": direction * magnitude,
                                       "firmware_echo_received": True})
        print("Arrived; firmware echoed command (not a physical angle measurement). Inspect clearance.")
    else:
        result["stop_reason"] = "requested_max_angle_reached"

    # The final candidate also needs explicit clearance confirmation.
    if current[AXES[axis]] != confirmed:
        key = ask("Final pose: Enter = confirm clear and return home; b = retreat/finish; anything else = hold/quit: ").strip().lower()
        if key == "":
            confirmed = float(current[AXES[axis]])
            result["last_confirmed_command_deg"] = confirmed
        elif key == "b":
            result["stop_reason"] = "operator_retreat_from_unconfirmed_pose"
        else:
            result["stop_reason"] = "operator_quit_hold"
            save()
            raise KeyboardInterrupt
        if key == "b":
            retreat = np.zeros(3)
            retreat[AXES[axis]] = confirmed
            transmit_move(ser, plan_move(platform, current, retreat, args))
            current = retreat
    result.pop("pending_command_deg", None)
    save()
    transmit_move(ser, plan_move(platform, current, np.zeros(3), args))
    result["returned_home"] = True
    save()
    print(f"Last operator-confirmed {axis} command: {confirmed:+.2f} deg; returned home.")


def main(argv=None):
    args = parse_args(argv)
    platform = make_platform()
    print("Platform angles are degrees; translations are held at ZERO relative to model home.")
    print("Geometry mm: base=66, platform=87.5, horn=45, rod=115.")
    try:
        model_results = preview(platform, args)
        plans = automatic_plans(platform, args, model_results) if args.auto else None
    except ValueError as error:
        raise SystemExit(f"Preflight rejected: {error}. No serial commands sent.") from error
    if not args.send and not args.diagnose:
        print("Preview only. These are MODEL limits, not measured clearance limits. Add --send to run.")
        return
    import serial

    filename = ROOT / "motion_capture" / f"range_test_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    log = {"settings": vars(args), "model_preview": model_results, "directions": [],
           "status": "started", "actual_platform_angles_measured": False}

    def save():
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    save()
    print(f"Report: {filename}")
    if args.diagnose:
        print("Read-only serial diagnosis at 38400 baud. No motion commands will be sent.")
    else:
        print("Start at calibrated HOME. q / Ctrl+C stops sending and HOLDS; neither cuts servo power.")
        print("Keep the servo power switch within reach. " +
              ("Automatic motion; watch clearance continuously." if args.auto else
               "Every step requires an operator clearance check."))
    try:
        # Set control-line states before opening the port to reduce resets.
        with serial.Serial(port=None, baudrate=38400, timeout=0.1, write_timeout=0.5,
                           dsrdtr=False, rtscts=False) as ser:
            ser.dtr = False
            ser.rts = False
            ser.port = args.port
            ser.open()
            # Listen immediately: discarding startup output can lose useful
            # diagnostics or a one-time firmware response.
            current = read_angles(ser, timeout=8.0)
            log["startup_firmware_angles"] = current.tolist()
            print("ESP32 Angles feedback:", current.tolist())
            if args.diagnose:
                log["status"] = "diagnosis_feedback_received_no_commands_sent"
                return
            home, _ = pose_angles(platform, np.zeros(3), args)
            if not np.allclose(current, home, atol=0.3, rtol=0):
                raise RuntimeError("Firmware targets are not at home. Return platform to calibrated home before testing.")
            if args.auto:
                run_automatic(ser, plans, log, save)
            else:
                axes = ["roll", "pitch"] if args.axis == "both" else [args.axis]
                for axis in axes:
                    for direction in (1, -1):
                        explore_direction(ser, platform, args, axis, direction, log, save)
        log["status"] = "completed"
    except (KeyboardInterrupt, EOFError):
        log["status"] = "operator_stopped_holding_pose"
        print("\nStopped sending. NO automatic return. Servos may still hold/load the mechanism.")
    except (RuntimeError, OSError, ValueError) as error:
        log["status"] = "error_stopped_holding_pose"
        log["error"] = str(error)
        print(f"\nSTOP: {error}\nNo automatic return was sent.")
    finally:
        save()
        print(f"Saved: {filename}")


if __name__ == "__main__":
    main()
