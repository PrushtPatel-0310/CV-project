"""Interactive safety-zone polygon drawing."""

from typing import Optional

import cv2
import numpy as np


PixelPoint = tuple[int, int]
PixelRectangle = tuple[int, int, int, int]
MINIMUM_ZONE_SIZE = 20


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

	def draw_status(
		self,
		frame: np.ndarray,
		contact_points: list[PixelPoint],
		inside_zone: bool,
		contour_count: int,
		candidate_count: int,
	) -> None:
		"""Draw safety status and concise edge-detector debug information."""
		status = "DANGER" if inside_zone else "SAFE"
		color = (0, 0, 255) if status == "DANGER" else (0, 180, 0)
		cv2.rectangle(frame, (10, 10), (520, 205), (0, 0, 0), -1)
		cv2.putText(
			frame,
			status,
			(25, 50),
			cv2.FONT_HERSHEY_SIMPLEX,
			1.2,
			color,
			3,
		)
		points_text = ", ".join(
			f"({point[0]}, {point[1]})" for point in contact_points
		)
		if not points_text:
			points_text = "None"
		debug_lines = (
			f"Zone: {'SELECTED' if self.selected else 'NOT SELECTED'}",
			f"Contours: {contour_count}  Candidates: {candidate_count}",
			f"Contact points: {points_text}",
			f"Inside zone: {'YES' if inside_zone else 'NO'}",
		)
		if not contact_points:
			debug_lines = ("NO CANDIDATE DETECTED",) + debug_lines
		for index, line in enumerate(debug_lines):
			cv2.putText(
				frame,
				line,
				(10, 85 + index * 20),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5,
				(255, 255, 255),
				1,
			)

	def draw(self, frame: np.ndarray) -> None:
		"""Draw the active drag rectangle or the selected danger zone."""
		rectangle = self._current_rectangle()
		if rectangle is not None:
			left, top, right, bottom = rectangle
			cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)

		if not self.selected:
			cv2.putText(
				frame,
				"Drag to select danger zone",
				(10, 175),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.7,
				(0, 255, 255),
				2,
			)
		else:
			cv2.putText(
				frame,
				"Danger zone selected",
				(10, 175),
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
