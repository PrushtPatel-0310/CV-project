"""Download the official MediaPipe Pose Landmarker model bundle."""

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIRECTORY / "pose_landmarker.task"
MODEL_URL = (
	"https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
	"pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


def download_model() -> Path:
	"""Download and verify the official model, returning its absolute path."""
	MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
	request = Request(MODEL_URL, headers={"User-Agent": "human-safety-zone-monitor"})

	try:
		with urlopen(request) as response:
			if not 200 <= response.status < 300:
				raise RuntimeError(
					f"MediaPipe model download returned HTTP status {response.status}."
				)

			with MODEL_PATH.open("wb") as output_file:
				while chunk := response.read(1024 * 1024):
					output_file.write(chunk)
	except HTTPError as error:
		if MODEL_PATH.exists():
			MODEL_PATH.unlink()
		raise RuntimeError(
			f"Failed to download MediaPipe Pose Landmarker model: "
			f"HTTP Error {error.code}: {error.reason}"
		) from error
	except (OSError, URLError, RuntimeError) as error:
		if MODEL_PATH.exists():
			MODEL_PATH.unlink()
		raise RuntimeError(f"Failed to download MediaPipe Pose Landmarker model: {error}") from error

	if not MODEL_PATH.is_file() or MODEL_PATH.stat().st_size == 0:
		raise RuntimeError("Download completed, but the model file is missing or empty.")

	return MODEL_PATH.resolve()


if __name__ == "__main__":
	try:
		model_path = download_model()
	except RuntimeError as error:
		print(error)
		raise SystemExit(1)

	print(f"Downloaded MediaPipe Pose Landmarker model to: {model_path}")
	print(f"File size: {model_path.stat().st_size} bytes")