import torch
from CNN_net import net, device
from dataset_load import test_loader
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print("运行test")
correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for j in range(10))
if __name__ == '__main__':
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # print(f'预测：{predicted},  实际标签：{labels}')
    print(f"correct:{correct}")
    print(f"total:{total}")
    print(f'accuracy:{correct / total}')
    for i in range(10):
        print(f'accuracy of {classes[i]}:{100*class_correct[i]/class_total[i]}%')
        print(f'{class_correct}      {class_total}')
