"""Application entry point for webcam capture and pose visualization."""

import cv2

from safety.zone import SafetyZone
from vision.pose_detector import PoseDetector, calculate_hip_center


WINDOW_TITLE = "Human Safety Zone Monitor"


def run_webcam(camera_index: int = 0) -> None:
	"""Capture webcam frames, draw pose landmarks, and quit on ``q``."""
	camera = cv2.VideoCapture(camera_index)

	if not camera.isOpened():
		print("Unable to open the webcam.")
		camera.release()
		return

	pose_detector = None
	safety_zone = SafetyZone()
	cv2.namedWindow(WINDOW_TITLE)
	cv2.setMouseCallback(WINDOW_TITLE, safety_zone.handle_mouse_event)
	try:
		pose_detector = PoseDetector()
		while True:
			frame_read, frame = camera.read()
			if not frame_read:
				print("Unable to read a frame from the webcam.")
				break

			if pose_detector is not None:
				poses = pose_detector.process_frame(frame)
				pose_detector.draw_landmarks(frame, poses)
				centers = []
				for pose_landmarks in poses:
					center = calculate_hip_center(
						pose_landmarks, frame.shape[1], frame.shape[0]
					)
					centers.append(center)
					pose_detector.draw_center(frame, center)
			else:
				centers = []
			current_center = next((center for center in centers if center is not None), None)
			inside_zone = any(safety_zone.contains_point(center) for center in centers)
			safety_zone.draw(frame)
			safety_zone.draw_status(frame, current_center, inside_zone)
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
		if pose_detector is not None:
			pose_detector.close()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	run_webcam()
