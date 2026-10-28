"""Interactive safety-zone polygon drawing."""

from typing import Any, Optional, Union

import cv2
import numpy as np

from config import TRACK_DANGER_CLEAR_FRAMES
from vision.tracker import TrackedCandidate

PixelPoint = tuple[int, int]
PixelRectangle = tuple[int, int, int, int]
BoundingBox = tuple[int, int, int, int]
MINIMUM_ZONE_SIZE = 20


def box_overlaps_zone(box: BoundingBox, zone: PixelRectangle) -> bool:
	"""Return whether candidate bounding box overlaps or touches danger zone."""
	x, y, width, height = box
	box_left = x
	box_top = y
	box_right = x + width
	box_bottom = y + height

	zone_left, zone_top, zone_right, zone_bottom = zone
	return (
		box_right >= zone_left
		and box_left <= zone_right
		and box_bottom >= zone_top
		and box_top <= zone_bottom
	)


class SafetyZone:
	"""Store and render one user-defined rectangular zone in pixel coordinates."""

	def __init__(self) -> None:
		self._drag_start: Optional[PixelPoint] = None
		self._drag_current: Optional[PixelPoint] = None
		self._rectangle: Optional[PixelRectangle] = None

	@property
	def selected(self) -> bool:
		"""Return whether a valid danger-zone rectangle has been selected."""
		return self._rectangle is not None

	@property
	def rectangle(self) -> Optional[PixelRectangle]:
		"""Return the selected rectangle as left, top, right, bottom pixels."""
		return self._rectangle

	def confirm(self) -> None:
		"""Confirm the current zone (keyboard shortcut support)."""
		pass

	def handle_mouse_event(
		self,
		event: int,
		x: int,
		y: int,
		flags: int = 0,
		userdata: Optional[object] = None,
	) -> None:
		"""Create a rectangle from a left-button drag in frame pixel coordinates."""
		if event == cv2.EVENT_LBUTTONDOWN and self._rectangle is None:
			self._drag_start = (x, y)
			self._drag_current = (x, y)
		elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
			if self._drag_start is not None:
				self._drag_current = (x, y)
		elif event == cv2.EVENT_LBUTTONUP:
			if self._drag_start is not None:
				self._drag_current = (x, y)
				candidate = self._normalized_rectangle(
					self._drag_start, self._drag_current
				)
				if candidate[2] - candidate[0] >= MINIMUM_ZONE_SIZE and candidate[3] - candidate[1] >= MINIMUM_ZONE_SIZE:
					self._rectangle = candidate
			self._drag_start = None
			self._drag_current = None

	def reset(self) -> None:
		"""Clear the selected rectangle and any active drag."""
		self._drag_start = None
		self._drag_current = None
		self._rectangle = None

	def contains_point(self, point: Optional[PixelPoint]) -> bool:
		"""Return whether a pixel point is inside or on the rectangle boundary."""
		if point is None or self._rectangle is None:
			return False

		left, top, right, bottom = self._rectangle
		return left <= point[0] <= right and top <= point[1] <= bottom

	def overlaps_box(self, box: BoundingBox) -> bool:
		"""Return whether a candidate bounding box overlaps or touches the selected zone."""
		if self._rectangle is None:
			return False
		return box_overlaps_zone(box, self._rectangle)

	def overlapping_candidates(
		self, candidates: list[BoundingBox]
	) -> list[BoundingBox]:
		"""Return candidate bounding boxes that overlap or touch the selected zone."""
		if not self.selected:
			return []
		return [box for box in candidates if self.overlaps_box(box)]

	def update_tracks_danger(
		self, tracks: list[TrackedCandidate]
	) -> list[TrackedCandidate]:
		"""Update danger status of each tracked candidate based on zone overlap."""
		if not self.selected:
			for track in tracks:
				track.in_danger = False
				track.danger_clear_count = 0
			return []

		in_danger_tracks: list[TrackedCandidate] = []
		for track in tracks:
			# Check both smoothed box and raw box so smoothing never hides a genuine overlap
			is_overlapping = self.overlaps_box(track.box) or self.overlaps_box(track.raw_box)

			if is_overlapping:
				track.in_danger = True
				track.danger_clear_count = 0
			else:
				if track.in_danger:
					track.danger_clear_count += 1
					if track.danger_clear_count >= TRACK_DANGER_CLEAR_FRAMES:
						track.in_danger = False
						track.danger_clear_count = 0

			if track.in_danger and track.confirmed:
				in_danger_tracks.append(track)

		return in_danger_tracks

	def evaluate_status(
		self, candidates_or_tracks: Union[list[TrackedCandidate], list[BoundingBox]]
	) -> str:
		"""Evaluate safety status based on zone selection and candidate/track overlaps."""
		if not self.selected:
			return "ZONE NOT SELECTED"
		if not candidates_or_tracks:
			return "NO CANDIDATE DETECTED"

		if isinstance(candidates_or_tracks[0], TrackedCandidate):
			confirmed_tracks = [t for t in candidates_or_tracks if t.confirmed]
			if not confirmed_tracks:
				return "NO CANDIDATE DETECTED"
			if any(t.in_danger for t in confirmed_tracks):
				return "DANGER"
			return "SAFE"

		if any(self.overlaps_box(box) for box in candidates_or_tracks):  # type: ignore[arg-type]
			return "DANGER"
		return "SAFE"

	def draw_status(
		self,
		frame: np.ndarray,
		tracks_or_candidates: Union[list[TrackedCandidate], list[BoundingBox]],
		arg3: Any = 0,
		arg4: int = 0,
		arg5: int = 0,
	) -> None:
		"""Draw safety status and candidate overlap debug information."""
		if tracks_or_candidates and isinstance(tracks_or_candidates[0], TrackedCandidate):
			tracks: list[TrackedCandidate] = tracks_or_candidates  # type: ignore[assignment]
			contour_count = int(arg3) if isinstance(arg3, (int, float)) else 0
			raw_candidate_count = int(arg4) if isinstance(arg4, (int, float)) else 0
			contact_points: list[PixelPoint] = []
		else:
			tracks = []
			raw_candidates: list[BoundingBox] = tracks_or_candidates  # type: ignore[assignment]
			contact_points = arg3 if isinstance(arg3, list) else []
			contour_count = int(arg4) if isinstance(arg4, (int, float)) else 0
			raw_candidate_count = int(arg5) if isinstance(arg5, (int, float)) else len(raw_candidates)

		if tracks:
			self.update_tracks_danger(tracks)
			status = self.evaluate_status(tracks)
			confirmed_tracks = [t for t in tracks if t.confirmed]
			in_danger_tracks = [t for t in confirmed_tracks if t.in_danger]
			any_danger = len(in_danger_tracks) > 0
		else:
			status = self.evaluate_status(raw_candidates if 'raw_candidates' in locals() else [])
			confirmed_tracks = []
			in_danger_tracks = []
			any_danger = status == "DANGER"

		if status == "DANGER":
			color = (0, 0, 255)
		elif status == "SAFE":
			color = (0, 180, 0)
		else:
			color = (0, 255, 255)

		font_scale = 1.2 if len(status) <= 6 else 0.85
		thickness = 3 if len(status) <= 6 else 2

		# Build debug lines
		debug_lines: list[str] = [
			f"Zone: {'SELECTED' if self.selected else 'NOT SELECTED'}",
			f"Contours: {contour_count}  Raw candidates: {raw_candidate_count}",
		]

		if tracks:
			debug_lines.append(f"Active tracks: {len(tracks)}  Confirmed: {len(confirmed_tracks)}")
			if confirmed_tracks:
				for track in confirmed_tracks:
					track_status = "DANGER" if track.in_danger else "SAFE"
					debug_lines.append(
						f"  ID #{track.track_id}: age={track.age} miss={track.misses} zone={track_status}"
					)
			else:
				debug_lines.append("  Track details: None (unconfirmed)")
		else:
			points_text = ", ".join(f"({p[0]}, {p[1]})" for p in contact_points) if contact_points else "None"
			debug_lines.append(f"Contact points: {points_text}")

		debug_lines.append(f"Zone overlap: {'YES' if any_danger else 'NO'}")

		# Dynamic HUD box sizing
		hud_height = max(205, 85 + len(debug_lines) * 20 + 10)
		self._hud_bottom = hud_height
		cv2.rectangle(frame, (10, 10), (550, hud_height), (0, 0, 0), -1)

		cv2.putText(
			frame,
			status,
			(25, 50),
			cv2.FONT_HERSHEY_SIMPLEX,
			font_scale,
			color,
			thickness,
		)

		for index, line in enumerate(debug_lines):
			cv2.putText(
				frame,
				line,
				(10, 80 + index * 20),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.48,
				(255, 255, 255),
				1,
			)

	def draw(self, frame: np.ndarray) -> None:
		"""Draw the active drag rectangle or the selected danger zone."""
		rectangle = self._current_rectangle()
		if rectangle is not None:
			left, top, right, bottom = rectangle
			cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)

		hint_text = (
			"Danger zone selected" if self.selected else "Drag to select danger zone"
		)
		hint_y = getattr(self, "_hud_bottom", 205) + 25
		cv2.putText(
			frame,
			hint_text,
			(10, hint_y),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.7,
			(0, 255, 255),
			2,
		)

	def _current_rectangle(self) -> Optional[PixelRectangle]:
		if self._drag_start is not None and self._drag_current is not None:
			return self._normalized_rectangle(self._drag_start, self._drag_current)
		return self._rectangle

	@staticmethod
	def _normalized_rectangle(
		start: PixelPoint, current: PixelPoint
	) -> PixelRectangle:
		return (
			min(start[0], current[0]),
			min(start[1], current[1]),
			max(start[0], current[0]),
			max(start[1], current[1]),
		)
