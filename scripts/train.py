from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torchvision.models import mobilenet_v2


def main() -> None:
	device = torch.device("cpu")

	repo_root = Path(__file__).resolve().parents[1]
	data_dir = repo_root / "data"
	model_dir = repo_root / "models"
	model_dir.mkdir(parents=True, exist_ok=True)

	transform = transforms.Compose(
		[
			transforms.Resize((64, 64)),
			transforms.ToTensor(),
		]
	)

	trainset = torchvision.datasets.CIFAR10(
		root=str(data_dir),
		train=True,
		download=True,
		transform=transform,
	)
	trainset = Subset(trainset, range(5000))
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)

	model = mobilenet_v2(weights=None)
	model.classifier[1] = torch.nn.Linear(model.last_channel, 10)
	model.to(device)

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	model.train()
	for epoch in range(1):
		running_loss = 0.0
		for images, labels in trainloader:
			images, labels = images.to(device), labels.to(device)

			outputs = model(images)
			loss = criterion(outputs, labels)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			running_loss += loss.item()

		avg_loss = running_loss / len(trainloader)
		print(f"Epoch {epoch + 1}/1 - Loss: {avg_loss:.4f}")

	model_path = model_dir / "baseline.pth"
	torch.save(model.state_dict(), model_path)
	print(f"Baseline model saved to: {model_path}")


if __name__ == "__main__":
	main()


