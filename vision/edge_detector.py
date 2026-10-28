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
	EDGE_MERGE_ENABLED,
	EDGE_MERGE_MAX_ASPECT_RATIO,
	EDGE_MERGE_MAX_VERTICAL_GAP,
	EDGE_MERGE_MIN_ASPECT_RATIO,
	EDGE_MERGE_MIN_HORIZONTAL_OVERLAP_RATIO,
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
		self.raw_candidate_count = 0
		self.candidate_count = 0
		self.raw_candidates: list[BoundingBox] = []
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
		self.raw_candidate_count = len(filtered_candidates)
		self.raw_candidates = list(filtered_candidates)

		if EDGE_MERGE_ENABLED and len(filtered_candidates) > 1:
			merged = self._merge_fragmented_boxes(filtered_candidates)
		else:
			merged = filtered_candidates

		candidates = self._suppress_overlaps(merged)
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
	def draw_candidate(
		frame: np.ndarray,
		box: BoundingBox,
		label: str = "Candidate",
		color: tuple[int, int, int] = (0, 255, 255),
		thickness: int = 2,
	) -> None:
		"""Draw one candidate box without labeling it as a person."""
		x, y, width, height = box
		cv2.rectangle(frame, (x, y), (x + width, y + height), color, thickness)
		cv2.putText(
			frame,
			label,
			(x, max(20, y - 8)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,
			color,
			thickness,
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

	@staticmethod
	def _merge_fragmented_boxes(boxes: list[BoundingBox]) -> list[BoundingBox]:
		"""Merge vertically aligned or overlapping fragmented candidate boxes."""
		if len(boxes) <= 1:
			return boxes

		num_boxes = len(boxes)
		parent = list(range(num_boxes))

		def find(i: int) -> int:
			path = []
			while parent[i] != i:
				path.append(i)
				i = parent[i]
			for node in path:
				parent[node] = i
			return i

		def union(i: int, j: int) -> None:
			root_i = find(i)
			root_j = find(j)
			if root_i != root_j:
				parent[root_i] = root_j

		for i in range(num_boxes):
			for j in range(i + 1, num_boxes):
				if EdgeCandidateDetector._can_merge(boxes[i], boxes[j]):
					union(i, j)

		groups: dict[int, list[BoundingBox]] = {}
		for i in range(num_boxes):
			root = find(i)
			groups.setdefault(root, []).append(boxes[i])

		merged_results: list[BoundingBox] = []
		for group in groups.values():
			composite = group[0]
			for other in group[1:]:
				composite = EdgeCandidateDetector._combine_boxes(composite, other)
			merged_results.append(composite)

		return merged_results

	@staticmethod
	def _can_merge(first: BoundingBox, second: BoundingBox) -> bool:
		"""Return whether two candidate boxes geometrically support merging."""
		x1, y1, w1, h1 = first
		x2, y2, w2, h2 = second

		first_left, first_right = x1, x1 + w1
		first_top, first_bottom = y1, y1 + h1

		second_left, second_right = x2, x2 + w2
		second_top, second_bottom = y2, y2 + h2

		# Check horizontal overlap
		h_overlap = max(0, min(first_right, second_right) - max(first_left, second_left))
		min_width = min(w1, w2)
		h_overlap_ratio = h_overlap / min_width if min_width > 0 else 0.0

		# Check vertical gap
		if first_bottom < second_top:
			v_gap = second_top - first_bottom
		elif second_bottom < first_top:
			v_gap = first_top - second_bottom
		else:
			v_gap = 0

		# Substantial vertical alignment & proximity (e.g. torso + legs or head + torso)
		if (
			h_overlap_ratio >= EDGE_MERGE_MIN_HORIZONTAL_OVERLAP_RATIO
			and v_gap <= EDGE_MERGE_MAX_VERTICAL_GAP
		):
			comp_w = max(first_right, second_right) - min(first_left, second_left)
			comp_h = max(first_bottom, second_bottom) - min(first_top, second_top)
			if comp_w <= 0:
				return False
			comp_ratio = comp_h / comp_w
			return (
				EDGE_MERGE_MIN_ASPECT_RATIO
				<= comp_ratio
				<= EDGE_MERGE_MAX_ASPECT_RATIO
			)

		# Check if one box is substantially nested within another
		intersection = h_overlap * max(0, min(first_bottom, second_bottom) - max(first_top, second_top))
		smaller_area = min(w1 * h1, w2 * h2)
		if smaller_area > 0 and (intersection / smaller_area) >= 0.7:
			return True

		return False

	@staticmethod
	def _combine_boxes(first: BoundingBox, second: BoundingBox) -> BoundingBox:
		"""Compute the bounding union of two boxes."""
		x1, y1, w1, h1 = first
		x2, y2, w2, h2 = second
		left = min(x1, x2)
		top = min(y1, y2)
		right = max(x1 + w1, x2 + w2)
		bottom = max(y1 + h1, y2 + h2)
		return (left, top, right - left, bottom - top)
