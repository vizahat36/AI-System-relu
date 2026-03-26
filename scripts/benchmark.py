import csv
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2

INPUT_SIZE = 64


def evaluate_accuracy_torch(model: torch.nn.Module, dataloader, device: torch.device) -> float:
	model.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for images, labels in dataloader:
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			predicted = outputs.argmax(dim=1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	return 100.0 * correct / total


def evaluate_accuracy_onnx(session: ort.InferenceSession, dataloader) -> float:
	input_name = session.get_inputs()[0].name
	correct = 0
	total = 0
	for images, labels in dataloader:
		ort_inputs = {input_name: images.numpy().astype(np.float32)}
		outputs = session.run(None, ort_inputs)[0]
		predicted = np.argmax(outputs, axis=1)
		total += labels.size(0)
		correct += int((predicted == labels.numpy()).sum())
	return 100.0 * correct / total


def latency_torch(model: torch.nn.Module, iterations: int = 100) -> float:
	model.eval()
	with torch.no_grad():
		dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
		_ = model(dummy)  # warmup

		start = time.perf_counter()
		for _ in range(iterations):
			_ = model(dummy)
		end = time.perf_counter()

	return ((end - start) / iterations) * 1000.0


def latency_onnx(session: ort.InferenceSession, iterations: int = 100) -> float:
	input_name = session.get_inputs()[0].name
	dummy = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
	_ = session.run(None, {input_name: dummy})  # warmup

	start = time.perf_counter()
	for _ in range(iterations):
		_ = session.run(None, {input_name: dummy})
	end = time.perf_counter()

	return ((end - start) / iterations) * 1000.0


def main() -> None:
	device = torch.device("cpu")

	repo_root = Path(__file__).resolve().parents[1]
	data_dir = repo_root / "data"
	baseline_path = repo_root / "models" / "baseline.pth"
	onnx_path = repo_root / "models" / "model.onnx"
	output_csv = repo_root / "benchmarks" / "results.csv"
	output_csv.parent.mkdir(parents=True, exist_ok=True)

	if not baseline_path.exists():
		raise FileNotFoundError(f"Baseline model not found: {baseline_path}. Run train.py first.")
	if not onnx_path.exists():
		raise FileNotFoundError(f"ONNX model not found: {onnx_path}. Run optimize.py first.")

	transform = transforms.Compose(
		[
			transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
			transforms.ToTensor(),
		]
	)
	testset = torchvision.datasets.CIFAR10(
		root=str(data_dir),
		train=False,
		download=True,
		transform=transform,
	)
	testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

	baseline_model = mobilenet_v2(weights=None)
	baseline_model.classifier[1] = torch.nn.Linear(baseline_model.last_channel, 10)
	baseline_model.load_state_dict(torch.load(baseline_path, map_location=device))
	baseline_model.to(device)

	ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

	baseline_size_mb = baseline_path.stat().st_size / (1024 * 1024)
	optimized_size_mb = onnx_path.stat().st_size / (1024 * 1024)

	baseline_latency_ms = latency_torch(baseline_model)
	optimized_latency_ms = latency_onnx(ort_session)

	baseline_accuracy_pct = evaluate_accuracy_torch(baseline_model, testloader, device)
	optimized_accuracy_pct = evaluate_accuracy_onnx(ort_session, testloader)

	print(f"Baseline size (MB): {baseline_size_mb:.3f}")
	print(f"Optimized size (MB): {optimized_size_mb:.3f}")
	print(f"Baseline latency (ms/inference): {baseline_latency_ms:.3f}")
	print(f"Optimized latency (ms/inference): {optimized_latency_ms:.3f}")
	print(f"Baseline accuracy (%): {baseline_accuracy_pct:.2f}")
	print(f"Optimized accuracy (%): {optimized_accuracy_pct:.2f}")

	with output_csv.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["Metric", "Baseline", "Optimized"])
		writer.writerow(["Model size (MB)", f"{baseline_size_mb:.3f}", f"{optimized_size_mb:.3f}"])
		writer.writerow(["Inference latency (ms/inference)", f"{baseline_latency_ms:.3f}", f"{optimized_latency_ms:.3f}"])
		writer.writerow(["Accuracy (%)", f"{baseline_accuracy_pct:.2f}", f"{optimized_accuracy_pct:.2f}"])

	print(f"Benchmark saved to: {output_csv}")


if __name__ == "__main__":
	main()
