import cv2
import numpy as np
import os
import shutil

from keras import models, layers, backend
from models.vgg16 import VGG16
from tqdm import tqdm


def get_pairs():
    file_path = "../../data/data_a3/testPairs"

    with open(file_path) as f:
        content = f.readlines()

    pairs = []
    for line in content:
        index = line.find(' ')
        pairs.append([line[:index], line[index + 1:-1]])

    return pairs


vgg16 = VGG16(include_top=False,
              weights=None,
              input_shape=(224, 224, 3))

middle = models.Sequential()
middle.add(vgg16)

middle.add(layers.Flatten())
middle.add(layers.Dense(4096, activation='relu', name='fc1'))
middle.add(layers.Dense(4096, activation='relu', name='fc2'))
middle.add(layers.Lambda(lambda x: backend.l2_normalize(x, axis=1)))
middle.load_weights("../../weights/a3_best_model.hdf5", by_name=True)

cropped_test_dir_path = "../../data/data_a3/cropped_test"

pairs = get_pairs()

for name1, name2 in tqdm(pairs):
    image1 = cv2.imread(os.path.join(cropped_test_dir_path, name1))
    image2 = cv2.imread(os.path.join(cropped_test_dir_path, name2))

    embedding_original_1 = middle.predict(np.expand_dims(image1 / 255., axis=0))[0]
    embedding_hflip_1 = middle.predict(np.expand_dims(np.fliplr(image1) / 255., axis=0))[0]
    average_embedding_1 = (embedding_original_1 + embedding_hflip_1) / 2.

    embedding_original_2 = middle.predict(np.expand_dims(image2 / 255., axis=0))[0]
    embedding_hflip_2 = middle.predict(np.expand_dims(np.fliplr(image2) / 255., axis=0))[0]
    average_embedding_2 = (embedding_original_2 + embedding_hflip_2) / 2.

    dist = np.sum((average_embedding_1 - average_embedding_2) ** 2)
    dist = np.sqrt(dist)

    file = open("../../data/data_a3/results.txt", 'a')
    result = 1 - dist

    s = str(result) + '\n'
    file.write(s)
    file.close()
