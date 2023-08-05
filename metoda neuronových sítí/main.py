import cv2 as cv
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import math
from torchvision.transforms import Compose, ToTensor, Normalize, Resize


class CNN2(nn.Module):
    def __init__(self, n_feature, p=0.0):
        super(CNN2, self).__init__()
        self.n_feature = n_feature
        self.p = p
        # Creates the convolution layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_feature, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=n_feature, out_channels=n_feature, kernel_size=3)
        # Creates the linear layers
        # Where do this 5 * 5 come from?! Check it below
        self.fc1 = nn.Linear(n_feature * 5 * 5, 50)
        self.fc2 = nn.Linear(50, 8)
        # Creates dropout layers
        self.drop = nn.Dropout(self.p)

    def featurizer(self, x):
        # Featurizer
        # First convolutional block
        # 3@28x28 -> n_feature@26x26 -> n_feature@13x13
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        # Second convolutional block
        # n_feature * @13x13 -> n_feature@11x11 -> n_feature@5x5
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        # Input dimension (n_feature@5x5)
        # Output dimension (n_feature * 5 * 5)
        x = nn.Flatten()(x)
        return x

    def classifier(self, x):
        # Classifier
        # Hidden Layer
        # Input dimension (n_feature * 5 * 5)
        # Output dimension (50)
        if self.p > 0:
            x = self.drop(x)
        x = self.fc1(x)
        x = F.relu(x)
        # Output Layer
        # Input dimension (50)
        # Output dimension (3)
        if self.p > 0:
            x = self.drop(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        x = self.featurizer(x)
        x = self.classifier(x)
        return x

def classify(model, composer, img, classes):
    model = model.eval()
    img = cv.resize(img, (28, 28))
    image = Image.fromarray(img)
    image = composer(image).float()
    image = image.unsqueeze(0)

    output = model(image)
    _, predicted = torch.max(output.data,1)
    return classes[predicted.item()]



def drawDetections(img, bbox, hand_window_size_mult=1.6):
    Xupper = bbox[0] - 20 - math.ceil(hand_window_size_mult * bbox[2])
    Yupper = bbox[1] - math.ceil(((hand_window_size_mult - 1) / 2) * bbox[3])
    Xlower = bbox[0] - 20
    Ylower = bbox[1] + math.ceil(((hand_window_size_mult - 1) / 2 + 1) * bbox[3])
    cv.rectangle(img, [bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]], (0, 255, 0), 3)
    cv.rectangle(img, [Xupper, Yupper], [Xlower, Ylower], (0, 0, 255), 3)

    return Xupper, Yupper, Xlower, Ylower

def main():

    classes = ["Circle", "Five", "Four", "One", "Rocknroll", "Three", "Thumb", "Two"]
    model = torch.load("model.pt")

    mean = [0.7128, 0.6365, 0.5766]
    std= [0.1386, 0.1895, 0.2045]

    composer = Compose([Resize(28),
                        ToTensor(),
                        Normalize(torch.Tensor(mean),torch.Tensor(std))])

    mp_facedetector = mp.solutions.face_detection

    cap = cv.VideoCapture(0)


    with mp_facedetector.FaceDetection(min_detection_confidence=0.8) as face_detection:
        while cap.isOpened():
            success, img = cap.read()

            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            results = face_detection.process(img)

            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

            if results.detections:
                bbox = results.detections[0].location_data.relative_bounding_box

                h, w, c = img.shape

                boundingBox = [int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)]

                Xup, Yup, Xdown, Ydown = drawDetections(img, boundingBox, hand_window_size_mult=1.8)

                hand_cropped = img[np.abs(Yup + 5):np.abs(Ydown - 5), np.abs(Xup + 5):np.abs(Xdown - 5)]

                hand_cropped = cv.cvtColor(hand_cropped, cv.COLOR_BGR2RGB)


                if hand_cropped.any():

                    vysledneGesto = classify(model, composer, hand_cropped, classes)

                    cv.putText(img, vysledneGesto, (Xup + 10, Yup - 15), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                               cv.LINE_AA)

                    cv.imshow("img",img)
                    cv.waitKey(1)









if __name__ =="__main__":
    main()