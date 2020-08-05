import cv2
import numpy as np
import os

from keras import models, backend
from tqdm import tqdm


def get_files_paths(dir_path):
    file_path = "../../data/data_a2/testFiles.txt"

    with open(file_path) as f:
        content = f.readlines()

    files = []
    for line in content:
        files.append(os.path.join(dir_path, line)[:-1])

    return files


middle = models.load_model("../../weights/a2_best_model.hdf5", custom_objects={"backend": backend})

test_dir_path = "../../data/data_a2/test"
files = get_files_paths(test_dir_path)

for file in tqdm(files):
    image = cv2.imread(file)
    image = cv2.resize(image, (224, 224))
    embedding1 = middle.predict(np.expand_dims(image / 255., axis=0))[0][0]
    embedding2 = middle.predict(np.expand_dims(np.fliplr(image) / 255., axis=0))[0][0]

    prediction = (embedding1 + embedding2) / 2

    file = open("../../data/data_a2/results.txt", 'a')
    result = 1 - prediction
    line = str(result) + '\n'
    file.write(line)
    file.close()
