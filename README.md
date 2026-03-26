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

## Output Space (Fill Before Final Submission)
Add your final run outputs here.

### Training Output
```text
[Paste terminal output for: python scripts/train.py]
```

### Optimization Output
```text
[Paste terminal output for: python scripts/optimize.py]
```

### Benchmark Output
```text
[Paste terminal output for: python scripts/benchmark.py]
```

### Inference Output (Optimized)
```text
[Paste terminal output for: python run_inference.py --model optimized]
```

### Inference Output (Baseline)
```text
[Paste terminal output for: python run_inference.py --model baseline]
```

### Screenshots Space
- [Add screenshot: benchmark terminal output]
- [Add screenshot: results.csv]
- [Add screenshot: inference command output]

## Deliverables Checklist
- [x] GitHub repository with required structure
- [x] Baseline model training script
- [x] Optimization script (Quantization + ONNX)
- [x] Benchmark script with CSV output
- [x] Edge simulation (CPU, batch size 1)
- [x] Demo script (`run_inference.py --model optimized`)
- [ ] Final report document (`report.pdf`) completed with explanation and screenshots

## Author
- Name: [Your Name]
- Role: AI Systems Internship Candidate
