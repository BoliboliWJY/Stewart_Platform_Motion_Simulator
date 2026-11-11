import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_platform_range as explorer


class RangeChecks(unittest.TestCase):
    def setUp(self):
        self.platform = explorer.make_platform()
        self.args = explorer.parse_args(["--axis", "roll", "--max-angle", "1"])

    def test_home_mapping_matches_physical_pca_channels(self):
        angles, channels = explorer.pose_angles(self.platform, np.zeros(3), self.args)
        np.testing.assert_allclose(angles, np.full(6, 19.4))
        np.testing.assert_allclose(channels, [109.4, 70.6, 109.4, 70.6, 109.4, 70.6])

    def test_asymmetric_angles_match_firmware_channel_order_and_signs(self):
        # Distinct inputs catch reversal mistakes that the equal home angles hide.
        with patch.object(self.platform, "calculate", return_value=np.deg2rad([1, 2, 3, 4, 5, 6])):
            _, targets = explorer.pose_angles(self.platform, np.zeros(3), self.args)
        np.testing.assert_allclose(targets, [96, 85, 94, 87, 92, 89])

    def test_incident_command_is_blocked_before_any_serial_access(self):
        from unittest.mock import MagicMock

        serial_module = MagicMock()
        with patch.dict(sys.modules, {"serial": serial_module}), \
                patch.object(explorer, "make_platform") as make, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            explorer.main(["--port", "COM7", "--auto", "--max-angle", "90",
                           "--servo-min", "0", "--servo-max", "180", "--send"])
        serial_module.Serial.assert_not_called()
        make.assert_not_called()

    def test_full_move_is_checked_including_interior(self):
        def check_interior(platform, pose, args):
            if 0.4 < pose[0] < 0.6:
                raise ValueError("interior blocked")
            return np.zeros(6), np.full(6, 90.0)

        with patch.object(explorer, "pose_angles", side_effect=check_interior):
            with self.assertRaisesRegex(ValueError, "interior blocked"):
                explorer.plan_move(self.platform, np.zeros(3), np.array([1.0, 0, 0]), self.args)

    def test_quantized_commands_stay_within_window(self):
        frames = explorer.plan_move(self.platform, np.zeros(3), np.array([1.0, 0, 0]), self.args)
        self.assertGreaterEqual(len(frames), 20)
        self.assertAlmostEqual(frames[-1][0], 1.0)
        for _, angles in frames:
            targets = 90 + angles * explorer.SIGNS
            self.assertGreaterEqual(targets.min(), 60)
            self.assertLessEqual(targets.max(), 120)

    def test_nonfinite_ik_rejected(self):
        with patch.object(self.platform, "calculate", return_value=np.full(6, np.nan)):
            with self.assertRaisesRegex(ValueError, "no finite solution"):
                explorer.pose_angles(self.platform, np.zeros(3), self.args)

    def explore(self, keys, move_side_effect=None):
        log = {"directions": []}
        answers = iter(keys)
        moves = []

        def move(ser, frames):
            moves.append(frames)
            if move_side_effect is not None:
                move_side_effect()

        self.log = log
        self.moves = moves
        with contextlib.redirect_stdout(io.StringIO()), patch.object(explorer, "transmit_move", side_effect=move):
            explorer.explore_direction(None, self.platform, self.args, "roll", 1,
                                       log, lambda: None, ask=lambda prompt: next(answers))
        return log, moves

    def test_quit_does_not_return_or_mark_unconfirmed_pose_clear(self):
        with self.assertRaises(KeyboardInterrupt):
            self.explore(["", "q"])
        self.assertEqual(len(self.moves), 1)
        result = self.log["directions"][0]
        self.assertEqual(result["last_confirmed_command_deg"], 0)
        self.assertEqual(result["pending_command_deg"], 0.5)
        self.assertNotIn("returned_home", result)

    def test_interruption_during_move_sends_no_return(self):
        def stop():
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.explore([""], move_side_effect=stop)
        self.assertEqual(len(self.moves), 1)
        self.assertEqual(self.log["directions"][0]["last_confirmed_command_deg"], 0)

    def test_last_pose_requires_confirmation_and_is_recorded(self):
        log, moves = self.explore(["", "", ""])
        result = log["directions"][0]
        self.assertEqual(result["last_confirmed_command_deg"], 1)
        self.assertTrue(result["returned_home"])
        self.assertEqual(len(moves), 3)
        np.testing.assert_allclose(moves[-1][-1][1], np.full(6, 19.4))

    def test_retreat_excludes_unconfirmed_candidate(self):
        log, moves = self.explore(["", "b"])
        self.assertEqual(log["directions"][0]["last_confirmed_command_deg"], 0)
        np.testing.assert_allclose(moves[1][-1][1], np.full(6, 19.4))

    def test_stop_at_model_limit_never_transmits_rejected_pose(self):
        self.args.servo_min = 70.6
        self.args.servo_max = 109.4
        log, moves = self.explore([])
        self.assertIn("exceeds", log["directions"][0]["stop_reason"])
        for frames in moves:
            for _, angles in frames:
                np.testing.assert_allclose(angles, np.full(6, 19.4))

    def test_feedback_waits_for_matching_command(self):
        class Feedback:
            lines = iter([b"boot message\n", b"Angles: 1,2,3,4,5,6,\n",
                          b"Angles: 19.40,19.40,19.40,19.40,19.40,19.40,\n"])

            def readline(self):
                return next(self.lines)

        expected = np.full(6, 19.4)
        np.testing.assert_allclose(explorer.read_angles(Feedback(), expected=expected), expected)

    def test_partial_write_stops_before_next_frame(self):
        class PartialWrite:
            writes = 0

            def reset_input_buffer(self):
                pass

            def write(self, payload):
                self.writes += 1
                return len(payload) - 1

        ser = PartialWrite()
        with self.assertRaisesRegex(RuntimeError, "Incomplete serial write"):
            explorer.transmit_move(ser, [(0, np.full(6, 19.4)), (0.1, np.full(6, 19.5))])
        self.assertEqual(ser.writes, 1)

    def test_partial_feedback_line_survives_read_timeout(self):
        class Feedback:
            chunks = iter([b"Angles: 19.40,19.40,", b"", b"19.40,19.40,19.40,19.40,\n"])

            def readline(self):
                return next(self.chunks)

        expected = np.full(6, 19.4)
        np.testing.assert_allclose(explorer.read_angles(Feedback(), expected=expected), expected)

    def test_no_bytes_has_distinct_diagnostic(self):
        with patch.object(explorer.time, "monotonic", side_effect=[0, 0, 6]):
            with self.assertRaisesRegex(RuntimeError, "received 0 bytes"):
                explorer.read_angles(type("SilentPort", (), {"readline": lambda self: b""})())

    def test_unexpected_firmware_output_is_in_error(self):
        with patch.object(explorer.time, "monotonic", side_effect=[0, 0, 6]):
            with self.assertRaisesRegex(RuntimeError, "calibration firmware"):
                explorer.read_angles(type("OtherFirmware", (), {
                    "readline": lambda self: b"calibration firmware\n"
                })())

    def test_wrong_command_feedback_has_distinct_diagnostic(self):
        with patch.object(explorer.time, "monotonic", side_effect=[0, 0, 6]):
            with self.assertRaisesRegex(RuntimeError, "but expected"):
                explorer.read_angles(type("OldCommand", (), {
                    "readline": lambda self: b"Angles: 0,0,0,0,0,0,\n"
                })(), expected=np.full(6, 19.4))

    def test_diagnosis_sends_no_motion_commands(self):
        from unittest.mock import MagicMock

        serial_module = MagicMock()
        fake_port = serial_module.Serial.return_value.__enter__.return_value
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.dict(sys.modules, {"serial": serial_module}), \
                patch.object(explorer, "read_angles", return_value=np.zeros(6)), \
                patch.object(explorer, "explore_direction") as explore, \
                patch.object(explorer, "transmit_move") as transmit, \
                patch.object(explorer.Path, "write_text"), \
                patch.object(explorer.Path, "mkdir"):
            explorer.main(["--diagnose", "--max-angle", "0.5"])
        fake_port.write.assert_not_called()
        explore.assert_not_called()
        transmit.assert_not_called()

    def test_diagnose_and_send_are_mutually_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            explorer.parse_args(["--diagnose", "--send"])

    def test_nonfinite_settings_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for flag in ("--speed", "--step", "--servo-min", "--max-angle",
                         "--boundary-margin", "--servo-speed"):
                with self.subTest(flag=flag), self.assertRaises(SystemExit):
                    explorer.parse_args([flag, "nan"])

    def test_invalid_automatic_options_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for options in (["--boundary-margin", "0"], ["--servo-speed", "-1"],
                            ["--max-angle", "91"]):
                with self.subTest(options=options), self.assertRaises(SystemExit):
                    explorer.parse_args(options)

    def auto_plan(self):
        args = explorer.parse_args(["--auto", "--max-angle", "90",
                                    "--servo-min", "0", "--servo-max", "180"])
        results = [{"axis": "roll", "direction": 1, "last_model_pass_deg": 5}]
        with contextlib.redirect_stdout(io.StringIO()):
            plans = explorer.automatic_plans(self.platform, args, results)
        return args, plans

    def test_automatic_roundtrip_respects_margin_window_and_servo_rate(self):
        args, plans = self.auto_plan()
        self.assertEqual(plans[0]["target_deg"], 3)
        home, _ = explorer.pose_angles(self.platform, np.zeros(3), args)
        previous = home
        for leg in ("outbound", "return"):
            previous_time = 0
            for offset, angles in plans[0][leg]:
                targets = 90 + angles * explorer.SIGNS
                self.assertTrue(np.all((targets >= 0) & (targets <= 180)))
                rate = np.max(np.abs(angles - previous)) / (offset - previous_time)
                self.assertLessEqual(rate, args.servo_speed + 1e-9)
                previous, previous_time = angles, offset
        np.testing.assert_allclose(previous, home)

    def test_automatic_path_rejects_branch_jump_before_serial_open(self):
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(explorer, "plan_move", return_value=[(1, np.full(6, 30.0))]), \
                self.assertRaisesRegex(ValueError, "branch discontinuity"):
            self.auto_plan()

    def run_auto(self, move_effect=None, wait_effect=None):
        _, plans = self.auto_plan()
        self.auto_log = {"directions": []}
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(explorer, "wait_until", side_effect=wait_effect), \
                patch.object(explorer, "transmit_move", side_effect=move_effect) as transmit:
            self.auto_transmit = transmit
            explorer.run_automatic(None, plans, self.auto_log, lambda: None,
                                   check_stop=lambda: None)

    def test_automatic_completion_returns_without_prompts(self):
        with patch("builtins.input", side_effect=AssertionError("unexpected prompt")):
            self.run_auto()
        self.assertEqual(self.auto_transmit.call_count, 2)
        result = self.auto_log["directions"][0]
        self.assertTrue(result["returned_home"])
        self.assertFalse(result["operator_clearance_confirmed"])
        self.assertEqual(result["phase"], "completed")

    def test_automatic_countdown_stop_sends_nothing(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_auto(wait_effect=KeyboardInterrupt)
        self.auto_transmit.assert_not_called()

    def test_automatic_outbound_stop_or_feedback_error_sends_no_return(self):
        for error in (KeyboardInterrupt, RuntimeError("feedback lost")):
            with self.subTest(error=error), self.assertRaises((KeyboardInterrupt, RuntimeError)):
                self.run_auto(move_effect=error)
            self.assertEqual(self.auto_transmit.call_count, 1)
            self.assertFalse(self.auto_log["directions"][0]["returned_home"])

    def test_automatic_stop_at_edge_sends_no_return(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_auto(wait_effect=[None, KeyboardInterrupt])
        self.assertEqual(self.auto_transmit.call_count, 1)
        self.assertEqual(self.auto_log["directions"][0]["phase"], "holding_edge")

    def test_automatic_stop_during_return_does_not_retry(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_auto(move_effect=[None, KeyboardInterrupt])
        self.assertEqual(self.auto_transmit.call_count, 2)
        result = self.auto_log["directions"][0]
        self.assertEqual(result["phase"], "returning_home")
        self.assertFalse(result["returned_home"])

    def test_automatic_transmit_stop_before_write(self):
        from unittest.mock import MagicMock

        port = MagicMock()
        with self.assertRaises(KeyboardInterrupt):
            explorer.transmit_move(port, [(0, np.full(6, 19.4))],
                                   check_stop=lambda: (_ for _ in ()).throw(KeyboardInterrupt))
        port.write.assert_not_called()

    def test_feedback_wait_can_be_stopped(self):
        from unittest.mock import MagicMock

        port = MagicMock()
        with self.assertRaises(KeyboardInterrupt):
            explorer.read_angles(port, check_stop=lambda: (_ for _ in ()).throw(KeyboardInterrupt))
        port.readline.assert_not_called()


if __name__ == "__main__":
    unittest.main()
