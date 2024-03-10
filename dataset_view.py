import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils
from dataset_load import train_loader
from dataset_load import classes
print("运行view_dataset")
if __name__ == '__main__':
    def imshow(img):
        """

        :param img:
        :return:
        """
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images))

    print(" ".join('%5s' % classes[labels[j]] for j in range(4)))



