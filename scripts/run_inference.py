import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torchvision.models import mobilenet_v2

INPUT_SIZE = 64


def run_baseline(models_dir: Path) -> None:
	model_path = models_dir / "baseline.pth"
	if not model_path.exists():
		raise FileNotFoundError(f"Baseline model not found: {model_path}")

	model = mobilenet_v2(weights=None)
	model.classifier[1] = torch.nn.Linear(model.last_channel, 10)
	model.load_state_dict(torch.load(model_path, map_location="cpu"))
	model.eval()

	dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

	with torch.no_grad():
		start = time.perf_counter()
		output = model(dummy)
		end = time.perf_counter()

	prediction = int(output.argmax(dim=1).item())
	latency_ms = (end - start) * 1000.0

	print("Model: baseline (PyTorch)")
	print(f"CPU only: True")
	print(f"Batch size: 1")
	print(f"Prediction: {prediction}")
	print(f"Latency (ms): {latency_ms:.3f}")


def run_optimized(models_dir: Path) -> None:
	onnx_path = models_dir / "model.onnx"
	if not onnx_path.exists():
		raise FileNotFoundError(f"Optimized ONNX model not found: {onnx_path}")

	session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
	input_name = session.get_inputs()[0].name
	dummy = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)

	start = time.perf_counter()
	output = session.run(None, {input_name: dummy})[0]
	end = time.perf_counter()

	prediction = int(np.argmax(output, axis=1)[0])
	latency_ms = (end - start) * 1000.0

	print("Model: optimized (ONNX Runtime)")
	print(f"CPU only: True")
	print(f"Batch size: 1")
	print(f"Prediction: {prediction}")
	print(f"Latency (ms): {latency_ms:.3f}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Run single-sample inference for edge simulation.")
	parser.add_argument(
		"--model",
		choices=["baseline", "optimized"],
		default="optimized",
		help="Which model type to run.",
	)
	args = parser.parse_args()

	repo_root = Path(__file__).resolve().parents[1]
	models_dir = repo_root / "models"

	if args.model == "baseline":
		run_baseline(models_dir)
	else:
		run_optimized(models_dir)


if __name__ == "__main__":
	main()
