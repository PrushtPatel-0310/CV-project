"""MediaPipe Pose Landmarker integration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


PixelLandmark = tuple[int, int]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker.task"


@dataclass(frozen=True)
class DetectedLandmark:
	"""A pose landmark converted to pixels with its confidence metadata."""

	x: int
	y: int
	visibility: Optional[float]
	presence: Optional[float]


def calculate_hip_center(
	pose_landmarks: Sequence[DetectedLandmark],
	frame_width: int,
	frame_height: int,
	min_confidence: float = 0.5,
) -> Optional[PixelLandmark]:
	"""Return the visible hip midpoint in pixels, or ``None`` if unavailable."""
	left_hip_index = vision.PoseLandmark.LEFT_HIP.value
	right_hip_index = vision.PoseLandmark.RIGHT_HIP.value
	if len(pose_landmarks) <= right_hip_index:
		return None

	left_hip = pose_landmarks[left_hip_index]
	right_hip = pose_landmarks[right_hip_index]
	if not _landmark_is_reliable(left_hip, min_confidence):
		return None
	if not _landmark_is_reliable(right_hip, min_confidence):
		return None

	center_x = (left_hip.x + right_hip.x) / 2
	center_y = (left_hip.y + right_hip.y) / 2
	return (
		max(0, min(round(center_x), frame_width - 1)),
		max(0, min(round(center_y), frame_height - 1)),
	)


def _landmark_is_reliable(
	landmark: DetectedLandmark, min_confidence: float
) -> bool:
	"""Check that MediaPipe supplied sufficient visibility and presence confidence."""
	return (
		landmark.visibility is not None
		and landmark.presence is not None
		and landmark.visibility >= min_confidence
		and landmark.presence >= min_confidence
	)


class PoseDetector:
	"""Detect human pose landmarks and convert them to frame pixel coordinates."""

	def __init__(self, model_path: Optional[Union[str, Path]] = None) -> None:
		resolved_model_path = (
			Path(model_path).expanduser().resolve()
			if model_path is not None
			else DEFAULT_MODEL_PATH
		)
		if not resolved_model_path.is_file():
			raise FileNotFoundError(
				"MediaPipe Pose Landmarker model was not found at "
				f"{resolved_model_path}. Run: "
				"python3 scripts/download_pose_model.py"
			)

		options = vision.PoseLandmarkerOptions(
			base_options=BaseOptions(model_asset_path=str(resolved_model_path)),
			running_mode=vision.RunningMode.IMAGE,
			num_poses=1,
		)
		self._landmarker = vision.PoseLandmarker.create_from_options(options)

	def process_frame(self, frame: np.ndarray) -> list[list[DetectedLandmark]]:
		"""Return each detected pose as pixel-coordinate landmark points."""
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		image = mp.Image(mp.ImageFormat.SRGB, rgb_frame)
		result = self._landmarker.detect(image)

		frame_height, frame_width = frame.shape[:2]
		return [
			[
				DetectedLandmark(
					*self._to_pixel_coordinates(
						landmark.x, landmark.y, frame_width, frame_height
					),
					visibility=landmark.visibility,
					presence=landmark.presence,
				)
				for landmark in pose_landmarks
			]
			for pose_landmarks in result.pose_landmarks
		]

	def draw_landmarks(
		self, frame: np.ndarray, poses: list[list[DetectedLandmark]]
	) -> None:
		"""Draw detected pose landmarks and MediaPipe's standard skeleton."""
		for pose_landmarks in poses:
			for landmark in pose_landmarks:
				cv2.circle(frame, (landmark.x, landmark.y), 4, (0, 255, 0), -1)

			for connection in vision.PoseLandmarksConnections.POSE_LANDMARKS:
				if connection.start >= len(pose_landmarks) or connection.end >= len(pose_landmarks):
					continue
				cv2.line(
					frame,
					(pose_landmarks[connection.start].x, pose_landmarks[connection.start].y),
					(pose_landmarks[connection.end].x, pose_landmarks[connection.end].y),
					(255, 0, 0),
					2,
				)

	def draw_center(
		self, frame: np.ndarray, center: Optional[PixelLandmark]
	) -> None:
		"""Draw the hip center and its pixel coordinates when available."""
		if center is None:
			return

		cv2.circle(frame, center, 8, (0, 0, 255), -1)
		cv2.putText(
			frame,
			f"Center: ({center[0]}, {center[1]})",
			(center[0] + 10, center[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.6,
			(0, 0, 255),
			2,
		)

	def close(self) -> None:
		"""Release MediaPipe resources."""
		self._landmarker.close()

	@staticmethod
	def _to_pixel_coordinates(
		normalized_x: float,
		normalized_y: float,
		frame_width: int,
		frame_height: int,
	) -> PixelLandmark:
		pixel_x = round(normalized_x * (frame_width - 1))
		pixel_y = round(normalized_y * (frame_height - 1))
		return (
			max(0, min(pixel_x, frame_width - 1)),
			max(0, min(pixel_y, frame_height - 1)),
		)
