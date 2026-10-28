"""Integration and stability tests for temporal tracking and safety zone monitoring."""

import unittest

from safety.zone import SafetyZone
from vision.edge_detector import EdgeCandidateDetector
from vision.tracker import LightweightTracker


class TestTemporalStabilityScenarios(unittest.TestCase):
	"""Test the 7 required scenarios from Phase E for temporal stability."""

	def setUp(self) -> None:
		self.tracker = LightweightTracker(
			iou_threshold=0.2,
			center_dist_threshold=0.75,
			alpha=0.4,
			confirmation_frames=3,
			max_missed_frames=4,
		)
		self.zone = SafetyZone()
		# Danger zone at [200, 200] to [400, 400]
		self.zone._rectangle = (200, 200, 400, 400)

	def test_scenario_1_stationary_person(self) -> None:
		"""Scenario 1: Stationary person with slight detection jitter over 20 frames."""
		# Person stands around (50, 50, 50, 120), jittering by +/- 2 pixels
		base_box = (50, 50, 50, 120)
		for frame_idx in range(20):
			jitter_x = 50 + (frame_idx % 3) - 1
			jitter_y = 50 + ((frame_idx + 1) % 3) - 1
			confirmed = self.tracker.update([(jitter_x, jitter_y, 50, 120)])
			if frame_idx >= 2:  # Confirmed after 3 frames
				self.assertEqual(len(confirmed), 1)
				self.assertEqual(confirmed[0].track_id, 1)

		# Check status remains stably SAFE (outside danger zone)
		status = self.zone.evaluate_status(self.tracker.active_tracks)
		self.assertEqual(status, "SAFE")
		self.assertEqual(self.tracker.active_tracks[0].age, 20)
		self.assertEqual(self.tracker.active_tracks[0].misses, 0)

	def test_scenario_2_slow_moving_person(self) -> None:
		"""Scenario 2: Person moving slowly across frames (3px displacement per frame)."""
		for frame_idx in range(15):
			x = 30 + frame_idx * 3
			confirmed = self.tracker.update([(x, 50, 50, 120)])
			if frame_idx >= 2:
				self.assertEqual(len(confirmed), 1)
				self.assertEqual(confirmed[0].track_id, 1)

		# Smoothed track box follows trajectory without dropping ID
		self.assertEqual(self.tracker.active_tracks[0].track_id, 1)
		self.assertGreater(self.tracker.active_tracks[0].box[0], 50)

	def test_scenario_3_fast_moving_person(self) -> None:
		"""Scenario 3: Person moving briskly (larger step matched by center distance)."""
		# Starts at x=50, jumps by 20px per frame
		for frame_idx in range(5):
			x = 50 + frame_idx * 20
			self.tracker.update([(x, 50, 50, 120)])

		# Track ID remains continuous and confirmed
		confirmed = self.tracker.confirmed_tracks
		self.assertEqual(len(confirmed), 1)
		self.assertEqual(confirmed[0].track_id, 1)

	def test_scenario_4_brief_occlusion_or_missed_detections(self) -> None:
		"""Scenario 4: Person temporarily occluded for 2 frames."""
		# 1. Establish confirmed track
		for _ in range(4):
			self.tracker.update([(50, 50, 50, 120)])

		track_id = self.tracker.confirmed_tracks[0].track_id

		# 2. Frame drop / occlusion for 2 frames
		self.tracker.update([])
		self.assertEqual(len(self.tracker.confirmed_tracks), 1)
		self.assertEqual(self.tracker.confirmed_tracks[0].misses, 1)

		self.tracker.update([])
		self.assertEqual(len(self.tracker.confirmed_tracks), 1)
		self.assertEqual(self.tracker.confirmed_tracks[0].misses, 2)

		# Status is still SAFE because confirmed track is preserved during brief miss
		status = self.zone.evaluate_status(self.tracker.active_tracks)
		self.assertEqual(status, "SAFE")

		# 3. Detection reappears
		self.tracker.update([(52, 50, 50, 120)])
		self.assertEqual(self.tracker.confirmed_tracks[0].track_id, track_id)
		self.assertEqual(self.tracker.confirmed_tracks[0].misses, 0)

	def test_scenario_5_empty_scene(self) -> None:
		"""Scenario 5: Empty scene produces NO CANDIDATE DETECTED, never implies SAFE."""
		self.tracker.update([])
		status = self.zone.evaluate_status(self.tracker.active_tracks)
		self.assertEqual(status, "NO CANDIDATE DETECTED")

		# If zone is unselected, status is ZONE NOT SELECTED
		unselected_zone = SafetyZone()
		self.assertEqual(
			unselected_zone.evaluate_status(self.tracker.active_tracks),
			"ZONE NOT SELECTED",
		)

	def test_scenario_6_patterned_clothing_and_fragmented_contours(self) -> None:
		"""Scenario 6: Torso and legs split by clothing pattern merge before tracking."""
		# Broken contours from edge detector
		torso_fragment = (100, 80, 45, 60)
		legs_fragment = (102, 142, 42, 70)

		merged = EdgeCandidateDetector._merge_fragmented_boxes(
			[torso_fragment, legs_fragment]
		)
		self.assertEqual(len(merged), 1)

		# Tracker gets single composite candidate instead of two flickering candidates
		for _ in range(3):
			confirmed = self.tracker.update(merged)

		self.assertEqual(len(confirmed), 1)
		self.assertEqual(confirmed[0].box[3], 132)

	def test_scenario_7_entering_and_leaving_danger_zone(self) -> None:
		"""Scenario 7: Person enters danger zone, persists through miss, and clears gracefully."""
		# Person approaches zone (zone is [200, 200, 400, 400])
		# Frame 1-3: Outside zone at (100, 250, 50, 100) -> SAFE
		for _ in range(3):
			self.tracker.update([(100, 250, 50, 100)])
			self.zone.update_tracks_danger(self.tracker.active_tracks)

		self.assertEqual(self.zone.evaluate_status(self.tracker.active_tracks), "SAFE")

		# Frame 4: Person enters danger zone (x=180..230, overlaps zone_left=200) -> DANGER
		self.tracker.update([(180, 250, 50, 100)])
		self.zone.update_tracks_danger(self.tracker.active_tracks)
		self.assertEqual(self.zone.evaluate_status(self.tracker.active_tracks), "DANGER")
		self.assertTrue(self.tracker.active_tracks[0].in_danger)

		# Frame 5: Brief 1-frame missed detection while in danger -> DANGER remains active!
		self.tracker.update([])
		self.zone.update_tracks_danger(self.tracker.active_tracks)
		self.assertEqual(self.zone.evaluate_status(self.tracker.active_tracks), "DANGER")

		# Frame 6: Person steps out of danger zone (x=100..150)
		# Needs TRACK_DANGER_CLEAR_FRAMES (3) consecutive clean frames before clearing
		self.tracker.update([(100, 250, 50, 100)])
		self.zone.update_tracks_danger(self.tracker.active_tracks)
		self.assertEqual(self.zone.evaluate_status(self.tracker.active_tracks), "DANGER")

		# Frame 7: Second clean frame -> still DANGER (hysteresis)
		self.tracker.update([(100, 250, 50, 100)])
		self.zone.update_tracks_danger(self.tracker.active_tracks)
		self.assertEqual(self.zone.evaluate_status(self.tracker.active_tracks), "DANGER")

		# Frame 8: Third clean frame -> now cleared to SAFE!
		self.tracker.update([(100, 250, 50, 100)])
		self.zone.update_tracks_danger(self.tracker.active_tracks)
		self.assertEqual(self.zone.evaluate_status(self.tracker.active_tracks), "SAFE")
		self.assertFalse(self.tracker.active_tracks[0].in_danger)


if __name__ == "__main__":
	unittest.main()
