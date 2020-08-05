import os
import shutil
import cv2
import logging

from tqdm import tqdm

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.NOTSET
)


def copy_images_to_dir(source_path, target_path):
    """
    Copy images from the same class to one directory with the class name
    (because VGG16 interprets each directory as one class and all images in this directory are from that class)
    """

    files = sorted(os.listdir(source_path))
    for file in tqdm(files):
        name = file[:file.rfind("_")]
        directory = target_path + name
        if not os.path.isdir(directory):
            os.mkdir(directory)
        shutil.copyfile(source_path + file, os.path.join(directory, file))


def remove_dirs_with_one_image(path):
    dirs = os.listdir(path)
    for dir in tqdm(dirs):
        dir_path = os.path.join(path, dir)
        if not len(os.listdir(dir_path)) > 1:
            shutil.rmtree(dir_path)


def crop_images(path):
    dirs = [os.path.join(path, d) for d in os.listdir(path)]

    for dir in tqdm(dirs):
        files = os.listdir(dir)
        for file in files:
            img_path = os.path.join(dir, file)
            image = cv2.imread(img_path)
            crop_img = image[13:13 + 224, 13:13 + 224]
            cv2.imwrite(img_path, crop_img)


def add_empty_folders(source_path, target_path):
    """
    Adding empty folders to target_path directory,
    if there is no folder with the same name as in source_path directory.
    Because VGG16 interprets each folder as a class and there can not be the different amount of
    classes in training and validation set.
    """
    source_dirs = os.listdir(source_path)
    target_dirs = os.listdir(target_path)
    for dir in tqdm(source_dirs):
        if dir not in target_dirs:
            os.makedirs(os.path.join(target_path, dir))


def main():
    training_paths = ("../../data/data_a3/train/", "../../data/data_a3/cropped_train/")
    validation_paths = ("../../data/data_a3/validation/", "../../data/data_a3/cropped_validation/")
    paths = [training_paths, validation_paths]

    for source_path, target_path in paths:
        os.makedirs(target_path, exist_ok=True)

        logging.info("Copying images from the same class to one directory with the class name")
        copy_images_to_dir(source_path, target_path)

        logging.info("Removing directories with one image")
        remove_dirs_with_one_image(target_path)

        logging.info("Cropping images")
        crop_images(target_path)

    logging.info("Adding empty folders to validation directory")
    add_empty_folders(training_paths[1], validation_paths[1])

    logging.info("Adding empty folders to train directory")
    add_empty_folders(validation_paths[1], training_paths[1])

    test_dir_path = "../../data/data_a3/test"
    cropped_test_dir_path = "../../data/data_a3/cropped_test"
    os.makedirs(cropped_test_dir_path, exist_ok=True)
    for file in os.listdir(test_dir_path):
        shutil.copy(os.path.join(test_dir_path, file), os.path.join(cropped_test_dir_path, file))

    files = os.listdir(cropped_test_dir_path)

    for file in files:
        img_path = os.path.join(cropped_test_dir_path, file)
        image = cv2.imread(img_path)
        crop_img = image[13:13 + 224, 13:13 + 224]
        cv2.imwrite(img_path, crop_img)

    original_files = os.listdir(cropped_test_dir_path)
    desired_size = 224
    for file in tqdm(original_files):
        im_pth = os.path.join(cropped_test_dir_path, file)
        im = cv2.imread(im_pth)
        old_size = im.shape[:2]  # old_size is in (height, width) format
        ratio = 1  # float(desired_size)/max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        # new_size should be in (width, height) format
        im = cv2.resize(im, (new_size[1], new_size[0]))
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)
        cv2.imwrite(os.path.join(cropped_test_dir_path, file), new_im)


if __name__ == "__main__":
    main()
