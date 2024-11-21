from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json
import torch
from torchvision import transforms
import torch.optim as optim
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import ResNet50V2
from os.path import exists


class CustomDataset(Dataset):
    def __init__(self, images_path, annotations_path):
        self.images_path = images_path
        self.annotations_path = annotations_path

        # List all images
        self.images = [file for file in os.listdir(images_path) if file.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_path, img_name)
        annotation_path = os.path.join(self.annotations_path, img_name.replace('.jpg', '.json'))

        if exists(annotation_path):

            # Load annotations
            with open(annotation_path, 'r') as file:
                annotations = json.load(file)


                image = cv2.imread(img_path)
                image = img_to_array(image)

                # Convert annotations to a tensor
                boxes = []
                labels = []
                ls_img_path = []
                ls_img = []

                for obj in annotations['objects']:
                    xmin = obj['bbox']['xmin']
                    ymin = obj['bbox']['ymin']
                    xmax = obj['bbox']['xmax']
                    ymax = obj['bbox']['ymax']
                    label = obj['label']
                    boxes.append((xmin, ymin, xmax, ymax))
                    labels.append(label)
                    ls_img_path.append(img_path)
                    ls_img.append(image)

                # Return image and its corresponding bounding boxes
                return boxes, labels, ls_img_path, ls_img


path_to_images = "C:\\Users\\mariu\\Desktop\\Hackadon\\Multi-Object Detection (MOD) - in dev\\images_train\\"
path_to_annotations = "C:\\Users\\mariu\\Desktop\\Hackadon\\Multi-Object Detection (MOD) - in dev\\mtsd_v2_fully_annotated\\annotations\\"

# Assuming you have loaded your annotations into 'json_data'
dataset = CustomDataset(
    images_path=path_to_images,
    annotations_path=path_to_annotations
)


# ResNet50V2 Model setup
resnet = ResNet50V2(include_top=False, weights="imagenet")
resnet.trainable = True  # or False, depending on whether you want to fine-tune

flatten = Flatten()(resnet.output)

# Bounding Box Head
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid", name='bounding_box')(bboxHead)

# Class Label Head
softmaxHead = Dense(512, activation="relu")(flatten)
softmaxHead = Dense(512, activation="relu")(softmaxHead)
softmaxHead = Dense(num_classes, activation="softmax", name='class_label')(softmaxHead)

# Combine into one model
model = Model(inputs=resnet.input, outputs=[bboxHead, softmaxHead])

# Compile the model
model.compile(
    loss={'class_label': 'categorical_crossentropy', 'bounding_box': 'mean_squared_error'},
    optimizer=Adam(lr=0.001)
)

# Prepare the TensorFlow data loader
train_data_loader = tf.data.Dataset.from_generator(lambda: dataset, output_types=(tf.float32, (tf.float32, tf.float32)))

# Training the model
model.fit(train_data_loader, epochs=20, steps_per_epoch=len(dataset) // dataset.batch_size)

# Save the model
model.save(
    'C:\\Users\\mariu\\Desktop\\Hackadon\\Multi-Object Detection (MOD) - in dev\\mtsd_v2_fully_annotated\\model\\'
)
