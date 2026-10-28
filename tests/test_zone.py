"""Unit tests for danger-zone bounding box overlap detection and safety status."""

import unittest

from safety.zone import SafetyZone, box_overlaps_zone


class TestDangerZoneOverlap(unittest.TestCase):
	"""Test candidate bounding box overlap with danger zone."""

	def setUp(self) -> None:
		# Define standard danger zone: left=100, top=100, right=200, bottom=200
		self.zone = (100, 100, 200, 200)
		self.safety_zone = SafetyZone()
		self.safety_zone._rectangle = self.zone

	def test_box_fully_outside(self) -> None:
		"""Box is completely outside the danger zone on each side."""
		# Outside to the left
		box_left = (10, 120, 40, 50)  # x=10..50 < zone_left=100
		self.assertFalse(box_overlaps_zone(box_left, self.zone))
		self.assertFalse(self.safety_zone.overlaps_box(box_left))

		# Outside to the right
		box_right = (250, 120, 40, 50)  # x=250..290 > zone_right=200
		self.assertFalse(box_overlaps_zone(box_right, self.zone))
		self.assertFalse(self.safety_zone.overlaps_box(box_right))

		# Outside above
		box_above = (120, 20, 50, 40)  # y=20..60 < zone_top=100
		self.assertFalse(box_overlaps_zone(box_above, self.zone))
		self.assertFalse(self.safety_zone.overlaps_box(box_above))

		# Outside below
		box_below = (120, 250, 50, 40)  # y=250..290 > zone_bottom=200
		self.assertFalse(box_overlaps_zone(box_below, self.zone))
		self.assertFalse(self.safety_zone.overlaps_box(box_below))

		# Outside top-left diagonal
		box_diagonal = (10, 10, 40, 40)
		self.assertFalse(box_overlaps_zone(box_diagonal, self.zone))
		self.assertFalse(self.safety_zone.overlaps_box(box_diagonal))

	def test_box_fully_inside(self) -> None:
		"""Box is completely inside the danger zone."""
		# x=120..170 within 100..200, y=120..170 within 100..200
		box_inside = (120, 120, 50, 50)
		self.assertTrue(box_overlaps_zone(box_inside, self.zone))
		self.assertTrue(self.safety_zone.overlaps_box(box_inside))

	def test_box_partially_overlapping_left_boundary(self) -> None:
		"""Box partially overlaps left boundary; contact point (90, 170) is outside."""
		# x=70..110 (overlaps zone_left=100), y=120..170
		box_left_overlap = (70, 120, 40, 50)
		self.assertTrue(box_overlaps_zone(box_left_overlap, self.zone))
		self.assertTrue(self.safety_zone.overlaps_box(box_left_overlap))

	def test_box_partially_overlapping_right_boundary(self) -> None:
		"""Box partially overlaps right boundary."""
		# x=180..220 (overlaps zone_right=200), y=120..170
		box_right_overlap = (180, 120, 40, 50)
		self.assertTrue(box_overlaps_zone(box_right_overlap, self.zone))
		self.assertTrue(self.safety_zone.overlaps_box(box_right_overlap))

	def test_box_partially_overlapping_top_boundary(self) -> None:
		"""Box partially overlaps top boundary."""
		# x=120..170, y=80..120 (overlaps zone_top=100)
		box_top_overlap = (120, 80, 50, 40)
		self.assertTrue(box_overlaps_zone(box_top_overlap, self.zone))
		self.assertTrue(self.safety_zone.overlaps_box(box_top_overlap))

	def test_box_partially_overlapping_bottom_boundary(self) -> None:
		"""Box partially overlaps bottom boundary; contact point (145, 220) is outside."""
		# x=120..170, y=180..220 (overlaps zone_bottom=200)
		box_bottom_overlap = (120, 180, 50, 40)
		self.assertTrue(box_overlaps_zone(box_bottom_overlap, self.zone))
		self.assertTrue(self.safety_zone.overlaps_box(box_bottom_overlap))

	def test_box_touching_boundary(self) -> None:
		"""Box touches each boundary (boundary sharing)."""
		# Touching left boundary: box_right == zone_left == 100
		box_touch_left = (50, 120, 50, 50)
		self.assertTrue(box_overlaps_zone(box_touch_left, self.zone))
		self.assertTrue(self.safety_zone.overlaps_box(box_touch_left))

		# Touching right boundary: box_left == zone_right == 200
		box_touch_right = (200, 120, 50, 50)
		self.assertTrue(box_overlaps_zone(box_touch_right, self.zone))
		self.assertTrue(self.safety_zone.overlaps_box(box_touch_right))

		# Touching top boundary: box_bottom == zone_top == 100
		box_touch_top = (120, 50, 50, 50)
		self.assertTrue(box_overlaps_zone(box_touch_top, self.zone))
		self.assertTrue(self.safety_zone.overlaps_box(box_touch_top))

		# Touching bottom boundary: box_top == zone_bottom == 200
		box_touch_bottom = (120, 200, 50, 50)
		self.assertTrue(box_overlaps_zone(box_touch_bottom, self.zone))
		self.assertTrue(self.safety_zone.overlaps_box(box_touch_bottom))

	def test_box_enclosing_entire_zone(self) -> None:
		"""Box is larger than and completely encloses the danger zone."""
		# x=50..250, y=50..250 encloses zone [100, 200] x [100, 200]
		box_enclosing = (50, 50, 200, 200)
		self.assertTrue(box_overlaps_zone(box_enclosing, self.zone))
		self.assertTrue(self.safety_zone.overlaps_box(box_enclosing))

	def test_safety_status_states(self) -> None:
		"""Test the four status states: unselected, no candidates, danger, and safe."""
		unselected_zone = SafetyZone()
		candidate = (120, 120, 50, 50)

		# State 1: Zone not selected
		self.assertEqual(
			unselected_zone.evaluate_status([candidate]),
			"ZONE NOT SELECTED",
		)
		self.assertEqual(
			unselected_zone.evaluate_status([]),
			"ZONE NOT SELECTED",
		)

		# State 2: Zone selected, but no candidates detected
		self.assertEqual(
			self.safety_zone.evaluate_status([]),
			"NO CANDIDATE DETECTED",
		)

		# State 3: Zone selected, candidates present, at least one overlaps zone -> DANGER
		overlapping_box = (70, 120, 40, 50)
		outside_box = (10, 120, 40, 50)
		self.assertEqual(
			self.safety_zone.evaluate_status([outside_box, overlapping_box]),
			"DANGER",
		)

		# State 4: Zone selected, candidates present, none overlap zone -> SAFE
		self.assertEqual(
			self.safety_zone.evaluate_status([outside_box]),
			"SAFE",
		)

	def test_overlapping_candidates_filtering(self) -> None:
		"""Test filtering list of candidates down to only overlapping boxes."""
		outside_box = (10, 120, 40, 50)
		overlapping_box = (70, 120, 40, 50)
		inside_box = (120, 120, 50, 50)

		result = self.safety_zone.overlapping_candidates(
			[outside_box, overlapping_box, inside_box]
		)
		self.assertEqual(result, [overlapping_box, inside_box])


if __name__ == "__main__":
	unittest.main()
