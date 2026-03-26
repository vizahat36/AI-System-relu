import warnings
from pathlib import Path

import torch
from torchvision.models import mobilenet_v2


def main() -> None:
	repo_root = Path(__file__).resolve().parents[1]
	models_dir = repo_root / "models"
	models_dir.mkdir(parents=True, exist_ok=True)

	baseline_path = models_dir / "baseline.pth"
	quantized_path = models_dir / "quantized.pth"
	onnx_path = models_dir / "model.onnx"

	if not baseline_path.exists():
		raise FileNotFoundError(f"Baseline model not found: {baseline_path}. Run train.py first.")

	# Rebuild the same model architecture used for training.
	model = mobilenet_v2(weights=None)
	model.classifier[1] = torch.nn.Linear(model.last_channel, 10)
	model.load_state_dict(torch.load(baseline_path, map_location="cpu"))
	model.eval()

	# Dynamic quantization is robust for quick CPU inference optimization.
	warnings.filterwarnings(
		"ignore",
		message="torch.ao.quantization is deprecated*",
		category=DeprecationWarning,
	)
	quantized_model = torch.quantization.quantize_dynamic(
		model,
		{torch.nn.Linear},
		dtype=torch.qint8,
	)

	torch.save(quantized_model.state_dict(), quantized_path)
	print(f"Quantized model saved to: {quantized_path}")

	dummy_input = torch.randn(1, 3, 64, 64)
	# Export float model to ONNX for maximum exporter/runtime compatibility.
	torch.onnx.export(
		model,
		dummy_input,
		str(onnx_path),
		export_params=True,
		opset_version=13,
		do_constant_folding=True,
		dynamo=False,
		input_names=["input"],
		output_names=["output"],
		dynamic_axes={
			"input": {0: "batch_size", 2: "height", 3: "width"},
			"output": {0: "batch_size"},
		},
	)
	print(f"ONNX model exported to: {onnx_path}")


if __name__ == "__main__":
	main()
