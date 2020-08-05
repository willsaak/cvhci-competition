from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)



def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)



def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255



def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)


def preprocess_image(normalize_lighting=False, min_value=0., max_value=1.):
    """
    Wrapper preprocessing function for input images to handle additional parameters
    :param normalize_lighting: whether to normalize image lighting through dividing each channel by its mean value
    :param min_value: minimum value for rescaling interval
    :param max_value: maximum value for rescaling interval
    :return: preprocessing function to pass in the ImageDataGenerator
    """
    if min_value > max_value:
        min_value, max_value = max_value, min_value
    rescaling_range = max_value - min_value

    def rescale(image):
        if image.max() > 1:
            # Project values interval on [0.0; 1.0]
            image = image / 255.
        # Project values on [min_value; max_value]
        image = image * rescaling_range + min_value
        return image

    def normalize_lighting_and_rescale(image):
        # Normalize image lighting (this results in an image with values from [0.0; 1.0])
        image = image / np.mean(image, axis=(0, 1))
        image = image / image.max()
        # Project values on [min_value; max_value]
        image = image * rescaling_range + min_value
        return image

    if normalize_lighting is True:
        return normalize_lighting_and_rescale
    else:
        return rescale


def preprocess_mask(mask):
    """
    Preprocessing function to pass in the ImageDataGenerator
    """
    # Project values interval on [0.0; 1.0]
    if mask.max() > 1:
        mask[mask <= 127.5] = 0.
        mask[mask > 127.5] = 1.
    else:
        mask[mask <= .5] = 0.
        mask[mask > .5] = 1.
    return mask


class ImageMaskGenerator:

    def __init__(
            self,
            directory,
            augmentation_args=None,
            image_preprocessing=None,
            mask_preprocessing=None,
            background_as_class=False,
            target_size=(256, 256),
            image_color_mode="grayscale",
            mask_color_mode="grayscale",
            image_subdirectory="image",
            mask_subdirectory="label",
            batch_size=2,
            shuffle=True,
            seed=None,
            save_to_dir=None,
            image_save_prefix="image_",
            mask_save_prefix="label_",
            save_format='png',
            follow_links=False,
            subset=None,
            interpolation='nearest'
    ):
        self.background_as_class = background_as_class
        self.batch_size = batch_size
        # If no seed is given, it should be initialized as the same for both data generators
        # in order to provide an image and its mask simultaneously
        self.seed = seed if seed is not None else np.random.randint(np.iinfo(np.int).min, np.iinfo(np.int).max)

        image_args = dict(augmentation_args) if augmentation_args is not None else {}
        image_args.update(preprocessing_function=image_preprocessing)
        mask_args = dict(augmentation_args) if augmentation_args is not None else {}
        mask_args.update(preprocessing_function=mask_preprocessing)
        self.image_generator = ImageDataGenerator(**image_args).flow_from_directory(
            directory,
            target_size=target_size,
            color_mode=image_color_mode,
            classes=[image_subdirectory],
            class_mode=None,
            batch_size=self.batch_size,
            shuffle=shuffle,
            seed=self.seed,
            save_to_dir=save_to_dir,
            save_prefix=image_save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation
        )
        self.mask_generator = ImageDataGenerator(**mask_args).flow_from_directory(
            directory,
            target_size=target_size,
            color_mode=mask_color_mode,
            classes=[mask_subdirectory],
            class_mode=None,
            batch_size=self.batch_size,
            shuffle=shuffle,
            seed=self.seed,
            save_to_dir=save_to_dir,
            save_prefix=mask_save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation
        )
        # bath_size in both generators is forced to be the same,
        # but there can be different number of samples in images and masks subdirectories
        if self.image_generator.samples != self.mask_generator.samples:
            raise ValueError("Different number of images and masks.")
        self.samples = self.image_generator.samples
        self.image_shape = self.image_generator.image_shape
        self.mask_shape = (self.mask_generator.image_shape[0], self.mask_generator.image_shape[1],
                           self.mask_generator.image_shape[2] + 1 if self.background_as_class is True else
                           self.mask_generator.image_shape[2])

    def __next__(self):
        images = next(self.image_generator)
        masks = next(self.mask_generator)
        if self.background_as_class is True:
            if masks.shape[-1] == 1:
                # If there is only one class, background is negation of it
                backgrounds = 1. - masks
            else:
                # If there are many classes, background is there where no one class is
                backgrounds = 1. - masks.max(axis=-1, keepdims=True)
            # Append background as first channel
            return images, np.append(backgrounds, masks, axis=-1)
        else:
            return images, masks