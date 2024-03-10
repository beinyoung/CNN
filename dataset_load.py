import torch
import torchvision
import torchvision.transforms as transforms

print("运行dataset_load")
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
train_set = torchvision.datasets.CIFAR10('./data', train=True, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=1)

test_set = torchvision.datasets.CIFAR10('./data', train=False, transform=transform, download=True)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=True, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
