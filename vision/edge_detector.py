"""Deterministic edge and contour filtering for human-shaped candidates."""

import cv2
import numpy as np

from config import (
	EDGE_CANNY_HIGH_THRESHOLD,
	EDGE_CANNY_LOW_THRESHOLD,
	EDGE_DILATION_ENABLED,
	EDGE_DILATION_ITERATIONS,
	EDGE_DILATION_KERNEL_SIZE,
	EDGE_GAUSSIAN_KERNEL_SIZE,
	EDGE_MAX_HEIGHT_WIDTH_RATIO,
	EDGE_MIN_BOUNDING_BOX_HEIGHT,
	EDGE_MIN_BOUNDING_BOX_WIDTH,
	EDGE_MIN_CONTOUR_AREA,
	EDGE_MIN_HEIGHT_WIDTH_RATIO,
	EDGE_MORPHOLOGY_ITERATIONS,
	EDGE_MORPHOLOGY_KERNEL_SIZE,
	EDGE_NMS_IOU_THRESHOLD,
)


BoundingBox = tuple[int, int, int, int]
PixelPoint = tuple[int, int]


class EdgeCandidateDetector:
	"""Find geometrically filtered contours that may resemble people."""

	def __init__(self) -> None:
		self.contour_count = 0
		self.candidate_count = 0
		self.debug_images: dict[str, np.ndarray] = {}
		self._morphology_kernel = cv2.getStructuringElement(
			cv2.MORPH_RECT, EDGE_MORPHOLOGY_KERNEL_SIZE
		)
		self._dilation_kernel = cv2.getStructuringElement(
			cv2.MORPH_RECT, EDGE_DILATION_KERNEL_SIZE
		)

	def detect(self, frame: np.ndarray) -> list[BoundingBox]:
		"""Return candidate boxes in frame pixels for one BGR frame."""
		if frame.ndim != 3 or frame.shape[2] != 3:
			raise ValueError("EdgeCandidateDetector expects a BGR color frame.")

		grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(grayscale, EDGE_GAUSSIAN_KERNEL_SIZE, 0)
		edges = cv2.Canny(
			blurred, EDGE_CANNY_LOW_THRESHOLD, EDGE_CANNY_HIGH_THRESHOLD
		)
		closed_edges = cv2.morphologyEx(
			edges,
			cv2.MORPH_CLOSE,
			self._morphology_kernel,
			iterations=EDGE_MORPHOLOGY_ITERATIONS,
		)
		processed_edges = closed_edges
		if EDGE_DILATION_ENABLED:
			processed_edges = cv2.dilate(
				closed_edges,
				self._dilation_kernel,
				iterations=EDGE_DILATION_ITERATIONS,
			)
		contours, _ = cv2.findContours(
			processed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
		)
		self.contour_count = len(contours)
		filtered_candidates = [
			box
			for contour in contours
			if self._is_candidate(contour, box := cv2.boundingRect(contour))
		]
		candidates = self._suppress_overlaps(filtered_candidates)
		self.candidate_count = len(candidates)
		self.debug_images = {
			"Original frame": frame.copy(),
			"Blurred grayscale": blurred,
			"Canny edges": edges,
			"Processed edges": processed_edges,
		}
		return candidates

	@staticmethod
	def contact_point(box: BoundingBox) -> PixelPoint:
		"""Return the bottom-center contact approximation for a candidate box."""
		x, y, width, height = box
		return (x + width // 2, y + height)

	@staticmethod
	def draw_candidate(frame: np.ndarray, box: BoundingBox) -> None:
		"""Draw one retained candidate box without labeling it as a person."""
		x, y, width, height = box
		cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 255), 2)
		cv2.putText(
			frame,
			"Candidate",
			(x, max(20, y - 8)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.55,
			(0, 255, 255),
			2,
		)

	@staticmethod
	def _is_candidate(
		contour: np.ndarray, box: BoundingBox
	) -> bool:
		area = cv2.contourArea(contour)
		_, _, width, height = box
		if (
			area < EDGE_MIN_CONTOUR_AREA
			or height < EDGE_MIN_BOUNDING_BOX_HEIGHT
			or width < EDGE_MIN_BOUNDING_BOX_WIDTH
		):
			return False
		if width <= 0:
			return False
		height_width_ratio = height / width
		return (
			EDGE_MIN_HEIGHT_WIDTH_RATIO
			<= height_width_ratio
			<= EDGE_MAX_HEIGHT_WIDTH_RATIO
		)

	@staticmethod
	def _suppress_overlaps(boxes: list[BoundingBox]) -> list[BoundingBox]:
		"""Keep the larger box when two contour boxes substantially overlap."""
		ordered_boxes = sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)
		selected: list[BoundingBox] = []
		for candidate in ordered_boxes:
			if all(
				EdgeCandidateDetector._intersection_over_union(candidate, existing)
				< EDGE_NMS_IOU_THRESHOLD
				for existing in selected
			):
				selected.append(candidate)
		return selected

	@staticmethod
	def _intersection_over_union(
		first: BoundingBox, second: BoundingBox
	) -> float:
		first_x, first_y, first_width, first_height = first
		second_x, second_y, second_width, second_height = second
		left = max(first_x, second_x)
		top = max(first_y, second_y)
		right = min(first_x + first_width, second_x + second_width)
		bottom = min(first_y + first_height, second_y + second_height)
		intersection = max(0, right - left) * max(0, bottom - top)
		first_area = first_width * first_height
		second_area = second_width * second_height
		union = first_area + second_area - intersection
		return intersection / union if union else 0.0
