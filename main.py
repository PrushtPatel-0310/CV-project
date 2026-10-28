"""Application entry point for webcam capture and edge candidates."""

import cv2

from config import EDGE_SHOW_DEBUG_VIEWS
from safety.zone import SafetyZone
from vision.edge_detector import EdgeCandidateDetector
from vision.tracker import LightweightTracker


WINDOW_TITLE = "Human Safety Zone Monitor"


def run_webcam(camera_index: int = 0) -> None:
	"""Capture webcam frames, draw candidates, and quit on ``q``."""
	camera = cv2.VideoCapture(camera_index)

	if not camera.isOpened():
		print("Unable to open the webcam.")
		camera.release()
		return

	detector = EdgeCandidateDetector()
	tracker = LightweightTracker()
	safety_zone = SafetyZone()
	show_raw_boxes = False
	cv2.namedWindow(WINDOW_TITLE)
	cv2.setMouseCallback(WINDOW_TITLE, safety_zone.handle_mouse_event)
	try:
		while True:
			frame_read, frame = camera.read()
			if not frame_read:
				print("Unable to read a frame from the webcam.")
				break

			frame = cv2.flip(frame, 1)
			candidates = detector.detect(frame)
			confirmed_tracks = tracker.update(candidates)

			# Optionally draw raw detected candidate boxes for debugging
			if show_raw_boxes:
				for raw_box in detector.raw_candidates:
					detector.draw_candidate(
						frame,
						raw_box,
						label="Raw",
						color=(180, 180, 180),
						thickness=1,
					)

			# Draw confirmed tracked candidate boxes and contact points
			for track in confirmed_tracks:
				track_color = (0, 0, 255) if track.in_danger else (0, 255, 255)
				status_label = f"ID #{track.track_id} [age:{track.age}]"
				detector.draw_candidate(
					frame,
					track.box,
					label=status_label,
					color=track_color,
					thickness=2,
				)
				contact_point = track.contact_point
				cv2.circle(frame, contact_point, 5, (255, 0, 255), -1)

			safety_zone.draw(frame)
			safety_zone.draw_status(
				frame,
				tracker.active_tracks,
				detector.contour_count,
				detector.raw_candidate_count,
			)

			if EDGE_SHOW_DEBUG_VIEWS:
				for debug_name, debug_image in detector.debug_images.items():
					cv2.imshow(debug_name, debug_image)
			cv2.imshow(WINDOW_TITLE, frame)

			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break
			if key == ord("r"):
				safety_zone.reset()
				tracker.reset()
			elif key == ord("d"):
				show_raw_boxes = not show_raw_boxes
			elif key in (10, 13):
				safety_zone.confirm()
	finally:
		camera.release()
		if EDGE_SHOW_DEBUG_VIEWS:
			for debug_name in detector.debug_images:
				cv2.destroyWindow(debug_name)
		cv2.destroyAllWindows()


if __name__ == "__main__":
	run_webcam()
