"""Application entry point for webcam capture and edge candidates."""

import cv2

from config import EDGE_SHOW_DEBUG_VIEWS
from safety.zone import SafetyZone
from vision.edge_detector import EdgeCandidateDetector


WINDOW_TITLE = "Human Safety Zone Monitor"


def run_webcam(camera_index: int = 0) -> None:
	"""Capture webcam frames, draw candidates, and quit on ``q``."""
	camera = cv2.VideoCapture(camera_index)

	if not camera.isOpened():
		print("Unable to open the webcam.")
		camera.release()
		return

	detector = EdgeCandidateDetector()
	safety_zone = SafetyZone()
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
			contact_points = []
			for candidate in candidates:
				detector.draw_candidate(frame, candidate)
				contact_point = detector.contact_point(candidate)
				contact_points.append(contact_point)
				cv2.circle(frame, contact_point, 5, (255, 0, 255), -1)
			inside_zone = any(safety_zone.contains_point(point) for point in contact_points)
			safety_zone.draw(frame)
			safety_zone.draw_status(
				frame,
				contact_points,
				inside_zone,
				detector.contour_count,
				detector.candidate_count,
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
