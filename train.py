import torch.optim as optim
from dataset_load import train_loader
import torch
import torch.nn as nn
from CNN_net import net
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("运行train")
if __name__ == '__main__':
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            # 正向传播
            outputs = net(inputs)
            # 计算损失值
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            # 损失值累加
            running_loss += loss.item()

            if i % 2000 == 1999:
                print('[%d,%d]loss:%.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print("training finished.")
