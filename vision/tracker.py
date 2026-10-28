"""Lightweight classical geometric multi-object tracker for candidate bounding boxes."""

from dataclasses import dataclass
import math
from typing import Optional

from config import (
	TRACK_CENTER_DIST_NORM_THRESHOLD,
	TRACK_CONFIRMATION_FRAMES,
	TRACK_IOU_MATCH_THRESHOLD,
	TRACK_MAX_MISSED_FRAMES,
	TRACK_SMOOTHING_ALPHA,
)

BoundingBox = tuple[int, int, int, int]
PixelPoint = tuple[int, int]


@dataclass
class TrackedCandidate:
	"""State of one tracked candidate bounding box across video frames."""

	track_id: int
	box: BoundingBox
	raw_box: BoundingBox
	hits: int = 1
	consecutive_hits: int = 1
	misses: int = 0
	age: int = 1
	confirmed: bool = False
	in_danger: bool = False
	danger_clear_count: int = 0

	@property
	def contact_point(self) -> PixelPoint:
		"""Return bottom-center contact point approximation."""
		x, y, width, height = self.box
		return (x + width // 2, y + height)


def calculate_iou(first: BoundingBox, second: BoundingBox) -> float:
	"""Compute Intersection over Union between two axis-aligned bounding boxes."""
	x1, y1, w1, h1 = first
	x2, y2, w2, h2 = second

	left = max(x1, x2)
	top = max(y1, y2)
	right = min(x1 + w1, x2 + w2)
	bottom = min(y1 + h1, y2 + h2)

	intersection = max(0, right - left) * max(0, bottom - top)
	area1 = w1 * h1
	area2 = w2 * h2
	union = area1 + area2 - intersection

	return intersection / union if union > 0 else 0.0


def calculate_normalized_center_distance(
	first: BoundingBox, second: BoundingBox
) -> float:
	"""Compute Euclidean distance between box centers normalized by box scale."""
	x1, y1, w1, h1 = first
	x2, y2, w2, h2 = second

	c1_x = x1 + w1 / 2.0
	c1_y = y1 + h1 / 2.0
	c2_x = x2 + w2 / 2.0
	c2_y = y2 + h2 / 2.0

	dist = math.hypot(c1_x - c2_x, c1_y - c2_y)
	scale = (math.hypot(w1, h1) + math.hypot(w2, h2)) / 2.0

	return dist / scale if scale > 0 else 1.0


def smooth_box(
	current: BoundingBox, previous: BoundingBox, alpha: float
) -> BoundingBox:
	"""Apply exponential moving average smoothing between previous and current box."""
	smoothed_x = alpha * current[0] + (1.0 - alpha) * previous[0]
	smoothed_y = alpha * current[1] + (1.0 - alpha) * previous[1]
	smoothed_w = alpha * current[2] + (1.0 - alpha) * previous[2]
	smoothed_h = alpha * current[3] + (1.0 - alpha) * previous[3]

	return (
		int(round(smoothed_x)),
		int(round(smoothed_y)),
		max(1, int(round(smoothed_w))),
		max(1, int(round(smoothed_h))),
	)


class LightweightTracker:
	"""Deterministic geometric tracker that associates and smooths candidate boxes."""

	def __init__(
		self,
		iou_threshold: float = TRACK_IOU_MATCH_THRESHOLD,
		center_dist_threshold: float = TRACK_CENTER_DIST_NORM_THRESHOLD,
		alpha: float = TRACK_SMOOTHING_ALPHA,
		confirmation_frames: int = TRACK_CONFIRMATION_FRAMES,
		max_missed_frames: int = TRACK_MAX_MISSED_FRAMES,
	) -> None:
		self.iou_threshold = iou_threshold
		self.center_dist_threshold = center_dist_threshold
		self.alpha = alpha
		self.confirmation_frames = confirmation_frames
		self.max_missed_frames = max_missed_frames

		self._next_track_id: int = 1
		self.tracks: list[TrackedCandidate] = []

	@property
	def active_tracks(self) -> list[TrackedCandidate]:
		"""Return all tracks currently maintained (both unconfirmed and confirmed)."""
		return [track for track in self.tracks if track.misses <= self.max_missed_frames]

	@property
	def confirmed_tracks(self) -> list[TrackedCandidate]:
		"""Return tracks that have satisfied the confirmation frame threshold."""
		return [track for track in self.active_tracks if track.confirmed]

	def reset(self) -> None:
		"""Clear all existing tracks and reset ID counter."""
		self.tracks.clear()
		self._next_track_id = 1

	def update(self, detections: list[BoundingBox]) -> list[TrackedCandidate]:
		"""Update tracker with candidate bounding boxes from the current frame."""
		matches, unmatched_track_indices, unmatched_detection_indices = (
			self._match(self.tracks, detections)
		)

		# 1. Update matched tracks
		for track_idx, det_idx in matches:
			track = self.tracks[track_idx]
			raw_box = detections[det_idx]

			# Exponential moving average smoothing
			track.box = smooth_box(raw_box, track.box, self.alpha)
			track.raw_box = raw_box
			track.hits += 1
			track.consecutive_hits += 1
			track.misses = 0
			track.age += 1

			if track.consecutive_hits >= self.confirmation_frames:
				track.confirmed = True

		# 2. Update unmatched tracks (missed detection in current frame)
		for track_idx in unmatched_track_indices:
			track = self.tracks[track_idx]
			track.misses += 1
			track.consecutive_hits = 0
			track.age += 1

		# 3. Create new tracks for unmatched detections
		for det_idx in unmatched_detection_indices:
			raw_box = detections[det_idx]
			is_confirmed = self.confirmation_frames <= 1
			new_track = TrackedCandidate(
				track_id=self._next_track_id,
				box=raw_box,
				raw_box=raw_box,
				hits=1,
				consecutive_hits=1,
				misses=0,
				age=1,
				confirmed=is_confirmed,
			)
			self._next_track_id += 1
			self.tracks.append(new_track)

		# 4. Remove tracks exceeding maximum missed frames
		self.tracks = [
			track for track in self.tracks if track.misses <= self.max_missed_frames
		]

		return self.confirmed_tracks

	def _match(
		self, tracks: list[TrackedCandidate], detections: list[BoundingBox]
	) -> tuple[list[tuple[int, int]], list[int], list[int]]:
		"""Associate tracks and detections 1-to-1 using IoU and center distance."""
		if not tracks:
			return [], [], list(range(len(detections)))
		if not detections:
			return [], list(range(len(tracks))), []

		# Compute pairwise match scores
		scored_pairs: list[tuple[float, int, int]] = []
		for t_idx, track in enumerate(tracks):
			for d_idx, det_box in enumerate(detections):
				iou = calculate_iou(track.box, det_box)
				norm_dist = calculate_normalized_center_distance(track.box, det_box)

				# Priority: IoU overlap first, then center distance gating
				if iou >= self.iou_threshold:
					score = 1.0 + iou
					scored_pairs.append((score, t_idx, d_idx))
				elif norm_dist <= self.center_dist_threshold:
					score = 1.0 - (norm_dist / self.center_dist_threshold)
					scored_pairs.append((score, t_idx, d_idx))

		# Sort by highest match score
		scored_pairs.sort(key=lambda item: item[0], reverse=True)

		# Greedy 1-to-1 assignment
		matched_tracks: set[int] = set()
		matched_detections: set[int] = set()
		matches: list[tuple[int, int]] = []

		for _, t_idx, d_idx in scored_pairs:
			if t_idx not in matched_tracks and d_idx not in matched_detections:
				matched_tracks.add(t_idx)
				matched_detections.add(d_idx)
				matches.append((t_idx, d_idx))

		unmatched_tracks = [
			i for i in range(len(tracks)) if i not in matched_tracks
		]
		unmatched_detections = [
			j for j in range(len(detections)) if j not in matched_detections
		]

		return matches, unmatched_tracks, unmatched_detections
