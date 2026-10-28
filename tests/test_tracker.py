"""Unit tests for lightweight tracker: matching, smoothing, confirmation, and expiry."""

import unittest

from vision.edge_detector import EdgeCandidateDetector
from vision.tracker import (
	LightweightTracker,
	TrackedCandidate,
	calculate_iou,
	calculate_normalized_center_distance,
	smooth_box,
)


class TestTrackerComponents(unittest.TestCase):
	"""Test core mathematical and geometric tracking utilities."""

	def test_iou_calculation(self) -> None:
		"""Test Intersection over Union calculation."""
		# Identical boxes -> IoU = 1.0
		box1 = (100, 100, 50, 50)
		self.assertAlmostEqual(calculate_iou(box1, box1), 1.0)

		# Completely separate boxes -> IoU = 0.0
		box2 = (200, 200, 50, 50)
		self.assertAlmostEqual(calculate_iou(box1, box2), 0.0)

		# 50% horizontal overlap (25x50 intersection = 1250, union = 2500 + 2500 - 1250 = 3750 -> 1/3)
		box3 = (125, 100, 50, 50)
		self.assertAlmostEqual(calculate_iou(box1, box3), 1250.0 / 3750.0)

	def test_normalized_center_distance(self) -> None:
		"""Test normalized center distance calculation."""
		# Identical boxes -> distance = 0
		box1 = (100, 100, 40, 80)
		self.assertAlmostEqual(calculate_normalized_center_distance(box1, box1), 0.0)

		# Displaced box
		box2 = (120, 100, 40, 80)  # Center shifted by 20px horizontally
		norm_dist = calculate_normalized_center_distance(box1, box2)
		self.assertGreater(norm_dist, 0.0)
		self.assertLess(norm_dist, 0.5)

	def test_exponential_moving_average_smoothing(self) -> None:
		"""Test coordinate smoothing using EMA."""
		previous = (100, 100, 50, 100)
		current = (120, 110, 60, 100)
		alpha = 0.4

		smoothed = smooth_box(current, previous, alpha)
		# expected x = 0.4*120 + 0.6*100 = 48 + 60 = 108
		# expected y = 0.4*110 + 0.6*100 = 44 + 60 = 104
		# expected w = 0.4*60 + 0.6*50 = 24 + 30 = 54
		# expected h = 0.4*100 + 0.6*100 = 100
		self.assertEqual(smoothed, (108, 104, 54, 100))


class TestLightweightTracker(unittest.TestCase):
	"""Test tracker lifecycle: association, confirmation, misses, and expiry."""

	def setUp(self) -> None:
		self.tracker = LightweightTracker(
			iou_threshold=0.2,
			center_dist_threshold=0.75,
			alpha=0.4,
			confirmation_frames=3,
			max_missed_frames=3,
		)

	def test_track_creation_and_confirmation(self) -> None:
		"""Track requires confirmation_frames before becoming confirmed."""
		detection = (100, 100, 50, 120)

		# Frame 1: Track created, not yet confirmed
		tracks_f1 = self.tracker.update([detection])
		self.assertEqual(len(self.tracker.active_tracks), 1)
		self.assertEqual(len(tracks_f1), 0)  # Not confirmed
		self.assertFalse(self.tracker.active_tracks[0].confirmed)
		track_id = self.tracker.active_tracks[0].track_id

		# Frame 2: Matched, still unconfirmed
		tracks_f2 = self.tracker.update([(102, 100, 50, 120)])
		self.assertEqual(len(tracks_f2), 0)
		self.assertFalse(self.tracker.active_tracks[0].confirmed)
		self.assertEqual(self.tracker.active_tracks[0].track_id, track_id)

		# Frame 3: Reaches confirmation_frames (3) -> Confirmed!
		tracks_f3 = self.tracker.update([(101, 101, 50, 120)])
		self.assertEqual(len(tracks_f3), 1)
		self.assertTrue(tracks_f3[0].confirmed)
		self.assertEqual(tracks_f3[0].track_id, track_id)

	def test_track_persistence_through_missed_detections(self) -> None:
		"""Confirmed track persists through missed frames until max_missed_frames."""
		detection = (100, 100, 50, 120)

		# Confirm the track over 3 frames
		for _ in range(3):
			self.tracker.update([detection])

		self.assertEqual(len(self.tracker.confirmed_tracks), 1)
		track_id = self.tracker.confirmed_tracks[0].track_id

		# Missed Frame 1: Detection drops
		self.tracker.update([])
		self.assertEqual(len(self.tracker.confirmed_tracks), 1)
		self.assertEqual(self.tracker.confirmed_tracks[0].misses, 1)

		# Missed Frame 2: Detection still missing
		self.tracker.update([])
		self.assertEqual(len(self.tracker.confirmed_tracks), 1)
		self.assertEqual(self.tracker.confirmed_tracks[0].misses, 2)

		# Detection returns in Frame 3 before expiry
		self.tracker.update([(102, 102, 50, 120)])
		self.assertEqual(len(self.tracker.confirmed_tracks), 1)
		self.assertEqual(self.tracker.confirmed_tracks[0].track_id, track_id)
		self.assertEqual(self.tracker.confirmed_tracks[0].misses, 0)

	def test_track_expiry(self) -> None:
		"""Track is removed after exceeding max_missed_frames."""
		detection = (100, 100, 50, 120)
		for _ in range(3):
			self.tracker.update([detection])

		# max_missed_frames is 3. Miss 4 frames:
		for _ in range(4):
			self.tracker.update([])

		self.assertEqual(len(self.tracker.active_tracks), 0)
		self.assertEqual(len(self.tracker.confirmed_tracks), 0)

	def test_one_to_one_matching_prevent_multiple_assignment(self) -> None:
		"""Ensure one detection cannot be assigned to multiple tracks."""
		# Spawn two distinct tracks
		self.tracker.update([(50, 50, 40, 100), (300, 50, 40, 100)])
		self.assertEqual(len(self.tracker.active_tracks), 2)

		# Next frame: provide only 1 detection close to first track
		self.tracker.update([(52, 51, 40, 100)])
		# Track 1 should match, Track 2 should have 1 miss
		matched_tracks = [t for t in self.tracker.active_tracks if t.misses == 0]
		unmatched_tracks = [t for t in self.tracker.active_tracks if t.misses == 1]
		self.assertEqual(len(matched_tracks), 1)
		self.assertEqual(len(unmatched_tracks), 1)

	def test_candidate_box_merging(self) -> None:
		"""Test fragmented vertical boxes merge while horizontally separate boxes do not."""
		# Torso and legs: vertically adjacent with significant horizontal overlap
		torso = (100, 100, 40, 60)
		legs = (105, 165, 38, 70)  # Gap of 5px, horizontal overlap ~33px / 38px > 0.35

		merged = EdgeCandidateDetector._merge_fragmented_boxes([torso, legs])
		self.assertEqual(len(merged), 1)
		comp_x, comp_y, comp_w, comp_h = merged[0]
		self.assertEqual(comp_x, 100)
		self.assertEqual(comp_y, 100)
		self.assertEqual(comp_h, 135)

		# Distant background contours: side-by-side, no horizontal overlap
		person = (100, 100, 40, 100)
		distant_object = (400, 100, 40, 100)
		not_merged = EdgeCandidateDetector._merge_fragmented_boxes([person, distant_object])
		self.assertEqual(len(not_merged), 2)


if __name__ == "__main__":
	unittest.main()
