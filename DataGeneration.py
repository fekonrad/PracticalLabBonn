import numpy as np
from numpy import random
import cv2
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class RandomShapes(Dataset):
    def __init__(self, num_images, height=512, width=512):
        self.num_images = num_images
        self.img, self.mask = generate_shapes(num_images, height, width)
        # Define a transformation to normalize the images
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])  # Assuming RGB images
        ])

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image = self.img[idx]
        image = self.transform(image)
        mask = torch.tensor(self.mask[idx])
        return image.to(torch.float32), mask.to(torch.float32)


def generate_shapes(num, height=512, width=512, background=False):
    """
    generates 'num' images of shape 'height' x 'width' with randomly placed shapes (triangles, rectangles, circles)
    returns images and corresponding binary masks for circular shapes
    """
    x = np.ndarray((num, height, width, 3))
    y = np.ndarray((num, height, width, 1))
    for i in range(num):
        img = np.zeros((height, width, 3))
        mask = np.zeros((height, width, 1))
        num_shapes = random.randint(0, 5)   # generates between 2 to 8 shapes per image
        for _ in range(num_shapes):
            shape_id = random.randint(0, 3)
            if shape_id == 0:   # rectangle
                x1 = random.randint(0, width - 1)
                y1 = random.randint(0, height - 1)
                x2 = random.randint(0, width - 1)
                y2 = random.randint(0, height - 1)

                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

            if shape_id == 1:   # triangle
                vertices = np.array([[random.randint(0, width-1), random.randint(0, height-1)],
                                     [random.randint(0, width-1), random.randint(0, height-1)],
                                     [random.randint(0, width-1), random.randint(0, height-1)]])
                vertices = vertices.reshape((-1, 1, 2))
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.fillPoly(img, [vertices], color)

            if shape_id == 2:   # circle
                center = (random.randint(0, width-1), random.randint(0, height-1))
                radius = random.randint(int(0.2*height), int(0.4*height))
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.circle(img, center, radius, color, -1)
                cv2.circle(mask, center, radius, (1, 1, 1), -1)  # binary mask corresponding to the circle

        img_rgb = cv2.cvtColor(cv2.convertScaleAbs(img), cv2.COLOR_BGR2RGB)
        x[i] = img_rgb
        y[i] = mask
    return x, y.transpose(0, 3, 1, 2)
"""the dataloader method above seems to transpose the shape of x automatically for some reason"""

""" Inspect a few generated images """
"""
while(True):
    x, y = generate_shapes(1)
    plt.clf()  # Clear the previous plot
    plt.subplot(1, 2, 1)
    plt.imshow(x[0].astype(int))
    plt.title('Input Image')

    plt.subplot(1, 2, 2)
    plt.imshow(y[0][0])
    plt.title('Output Mask')
    plt.tight_layout()
    plt.show()
    input("Press Enter to continue...")
"""
