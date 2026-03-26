# ReLU Technologies - AI Systems Internship Challenge

## Challenge Title
Efficient AI Inference for Edge Devices

## Problem Statement
This project focuses on optimizing a machine learning model for fast and efficient inference on resource-constrained hardware.

Goals:
- Reduce inference latency
- Reduce model size and memory footprint
- Maintain acceptable accuracy

## Dataset and Model
- Dataset: CIFAR-10
- Baseline model: MobileNetV2
- Device target: CPU (edge simulation)

## Optimization Techniques Used
This project applies at least two required optimization techniques:

1. Quantization
- Dynamic quantization applied to linear layers

2. ONNX optimization
- Exported model to ONNX
- Inference with ONNX Runtime (CPUExecutionProvider)

## Project Structure
```text
relu-inter-challenge/
|- data/
|- models/
|  |- baseline.pth
|  |- quantized.pth
|  |- model.onnx
|- scripts/
|  |- train.py
|  |- optimize.py
|  |- benchmark.py
|  |- run_inference.py
|- benchmarks/
|  |- results.csv
|- run_inference.py
|- requirements.txt
|- README.md
|- report.pdf
```

## Setup
Create and activate virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run Pipeline
Train baseline model:

```powershell
python scripts\train.py
```

Optimize model (quantized + ONNX export):

```powershell
python scripts\optimize.py
```

Benchmark baseline vs optimized:

```powershell
python scripts\benchmark.py
```

Run demo inference:

```powershell
python run_inference.py --model optimized
python run_inference.py --model baseline
```

## Edge Simulation (Task 4)
Simulation settings:
- CPU only
- Batch size = 1

Expected behavior:
- Lower latency for single-request inference
- Lower throughput compared to larger server-side batches
- Quantization and ONNX runtime improve speed with minimal or no accuracy change in this run

## Benchmark Results (Task 3)
Measured from `benchmarks/results.csv`:

| Metric | Baseline | Optimized |
|---|---:|---:|
| Model size (MB) | 8.762 | 8.506 |
| Inference latency (ms/inference) | 11.895 | 2.225 |
| Accuracy (%) | 26.17 | 26.17 |

## Output Space 
Add your final run outputs here.

### Training Output
```text
[(venv) PS E:\Relu-inter-challenge> python scripts\train.py
Epoch 1/1 - Loss: 2.1807
Baseline model saved to: E:\Relu-inter-challenge\models\baseline.pth]
```

### Optimization Output
```text
[(venv) PS E:\Relu-inter-challenge> python .\scripts\optimize.py
Quantized model saved to: E:\Relu-inter-challenge\models\quantized.pth
E:\Relu-inter-challenge\scripts\optimize.py:43: DeprecationWarning: You are using the legacy TorchScript-base export. Starting in PyTorch 2.9, the new torch.export-based ONNX exporter has become the default. For exporting control flow: httytorch.org/tutorials/beginner/onnx/export_control_flow_model_to_onnx_tutorial.html
torch.onnx.export(
ONNX model exported to: E:\Relu-inter-challenge\models\model.onnx]
```

### Benchmark Output
```text
[(venv) PS E:\Relu-inter-challenge> python .\scripts\benchmark.py
Baseline size (MB): 8.762
Optimized size (MB): 8.506
Baseline latency (ms/inference): 11.895
Optimized latency (ms/inference): 2.225
Baseline accuracy (%): 26.17
Optimized accuracy (%): 26.17
Benchmark saved to: E:\Relu-inter-challenge\benchmarks\results.csv]
```

### Inference Output (Optimized)
```text
[(venv) PS E:\Relu-inter-challenge> python run_inference.py --model optimized
Model: optimized (ONNX Runtime)
CPU only: True
Batch size: 1
Prediction: 1
Latency (ms): 10.368]
```

### Inference Output (Baseline)
```text
[(venv) PS E:\Relu-inter-challenge> python run_inference.py --model baseline
Model: baseline (PyTorch)
CPU only: True
Batch size: 1
Prediction: 1
Latency (ms): 304.463]
```

### Screenshots Space

- [train terminal output]
- ![WhatsApp Image 2026-03-26 at 9 54 30 PM](https://github.com/user-attachments/assets/00a5222e-f43a-40b3-ac9b-7879d66661c2)
- [optimization terminal output]
- <img width="670" height="124" alt="image" src="https://github.com/user-attachments/assets/d093bf75-aba8-4abe-b78c-600371af473e" />
 - [Add screenshot: benchmark terminal output]
-  <img width="462" height="125" alt="image" src="https://github.com/user-attachments/assets/bb737835-8e02-49fe-a37c-9e7736ecace4" />
- [results.csv ]
- <img width="391" height="110" alt="image" src="https://github.com/user-attachments/assets/0dc64bc4-6273-4739-916b-d3b81d8cbc39" />
- [inference command output ]
- <img width="475" height="89" alt="image" src="https://github.com/user-attachments/assets/57963023-2481-4f29-b94d-dcbd8ca6dd78" />
- ![inference command output]
- ![WhatsApp Image 2026-03-26 at 11 32 50 PM](https://github.com/user-attachments/assets/adb1a121-c460-4285-b380-7de23b8fdc4e)



## Author
- Name: [Mohammed Vijahath]
- Role: AI Systems Internship Candidate
