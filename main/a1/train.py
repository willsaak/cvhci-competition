import os
import datetime
import json
import random as rn
import numpy as np
import tensorflow as tf
from keras import backend, callbacks, optimizers
from keras.preprocessing.image import ImageDataGenerator

from models import unet
from models.custom_losses import image_binary_crossentropy, image_categorical_crossentropy
from models.custom_callbacks import TestPredictor
from main.a1.data import ImageMaskGenerator, preprocess_image, preprocess_mask

# Try to make the training reproducible (at least in terms of training data)
seed = 0
np.random.seed(seed)
rn.seed(seed)
tf.set_random_seed(seed)

# Setup allocating only as much GPU memory as needed
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
session = tf.Session(graph=tf.get_default_graph(), config=tf_config)
backend.tensorflow_backend.set_session(session)

# Preprocessing parameters
height, width, channels = 480, 640, 3
normalize_lighting = True
min_value, max_value = 0., 1.
background_as_class = True
augmentation_args = dict(
    rotation_range=36,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    # fill_mode='nearest',
    fill_mode='constant',
    cval=0.
)
# U-Net parameters
unet_args = dict(
    input_shape=(height, width, channels),
    classes=1,
    background_as_class=background_as_class,
    # up_conv="deconvolution",
    up_conv="upsampling",
    batch_normalization=False,
    # dropout_rate=(0., 0., 0., .5, .5, 0., 0., 0., 0.)
    dropout_rate=(0., .1, .3, .5, .5, .3, .1, 0., 0.)
)
# Training parameters
use_custom_losses = True
optimizer = optimizers.Adam(lr=1e-4)
data_dir = "data/data_a1/"
epochs = 100
train_batchsize = 2
# Other parameters
train_dir = os.path.join(data_dir, "train")
validation_dir = os.path.join(data_dir, "validation")
test_dir = os.path.join(data_dir, "test")
validation_batchsize = 1
test_batchsize = 1
image_preprocessing = preprocess_image(normalize_lighting=normalize_lighting, min_value=min_value, max_value=max_value)

run_name = "run_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
weights_dir = os.path.join("weights", run_name)
if not os.path.isdir(weights_dir):
    print("Create directory " + weights_dir + " for saving weights")
    os.makedirs(weights_dir)
results_dir = os.path.join("results", run_name)
if not os.path.isdir(results_dir):
    print("Create directory " + results_dir + " for saving results")
    os.makedirs(results_dir)
config = dict(height=height, width=width, channels=channels, normalize_lighting=normalize_lighting,
              min_value=min_value, max_value=max_value, background_as_class=background_as_class,
              augmentation_args=augmentation_args, unet_args=unet_args, use_custom_losses=use_custom_losses,
              optimizer=optimizers.serialize(optimizer), seed=seed, data_dir=data_dir, epochs=epochs,
              train_batchsize=train_batchsize)
with open(os.path.join(weights_dir, "config.json"), "w") as file:
    json.dump(config, file, indent=4)
with open(os.path.join(results_dir, "config.json"), "w") as file:
    json.dump(config, file, indent=4)

print("\nTrain dataset statistics:")
train_generator = ImageMaskGenerator(
    train_dir,
    augmentation_args=augmentation_args,
    image_preprocessing=image_preprocessing,
    mask_preprocessing=preprocess_mask,
    background_as_class=background_as_class,
    target_size=(height, width),
    image_color_mode="rgb",
    mask_color_mode="grayscale",
    image_subdirectory="image",
    mask_subdirectory="label",
    batch_size=train_batchsize,
    seed=seed
)
print("\nValidation dataset statistics:")
validation_generator = ImageMaskGenerator(
    validation_dir,
    augmentation_args=augmentation_args,
    image_preprocessing=image_preprocessing,
    mask_preprocessing=preprocess_mask,
    background_as_class=background_as_class,
    target_size=(height, width),
    image_color_mode="rgb",
    mask_color_mode="grayscale",
    image_subdirectory="image",
    mask_subdirectory="label",
    batch_size=validation_batchsize,
    seed=seed
)
print("\nTest dataset statistics:")
test_generator = ImageDataGenerator(preprocessing_function=image_preprocessing).flow_from_directory(
    test_dir,
    target_size=(height, width),
    color_mode='rgb',
    classes=["."],
    class_mode=None,
    batch_size=test_batchsize,
    shuffle=False
)

model = unet(**unet_args)
if use_custom_losses is True:
    loss = image_categorical_crossentropy if background_as_class is True else image_binary_crossentropy
else:
    loss = "categorical_crossentropy" if background_as_class is True else "binary_crossentropy"
model.compile(optimizer=optimizer, loss=loss, metrics=["acc"])
model.summary()

callbacks = [
    callbacks.ModelCheckpoint(os.path.join(weights_dir, 'best_loss.hdf5'), monitor="loss",
                              verbose=1, save_best_only=True, save_weights_only=False),
    callbacks.ModelCheckpoint(os.path.join(weights_dir, 'best_acc.hdf5'), monitor="acc",
                              verbose=1, save_best_only=True, save_weights_only=False),
    callbacks.ModelCheckpoint(os.path.join(weights_dir, 'best_val_loss.hdf5'), monitor="val_loss",
                              verbose=1, save_best_only=True, save_weights_only=False),
    callbacks.ModelCheckpoint(os.path.join(weights_dir, 'best_val_acc.hdf5'), monitor="val_acc",
                              verbose=1, save_best_only=True, save_weights_only=False),
    TestPredictor(test_generator, results_dir, save_prefix="out-", background_as_class=background_as_class)
]

print("\nTraining")
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / validation_generator.batch_size
)
model.save(os.path.join(weights_dir, 'last_epoch.hdf5'))
print("Train loss:", history.history["loss"])
print("Train acc:", history.history["acc"])
print("Train val loss:", history.history["val_loss"])
print("Train val acc:", history.history["val_acc"])

print("\nEvaluation")
evaluation = model.evaluate_generator(
    validation_generator,
    steps=10 * validation_generator.samples / validation_generator.batch_size,
    verbose=1
)
print("Evaluation loss:", evaluation[0])
print("Evaluation acc:", evaluation[1])

if use_custom_losses is True:
    best_loss_epoch, best_sum_loss = min(enumerate(history.history["loss"]), key=lambda x: x[1])
    best_val_loss_epoch, best_val_sum_loss = min(enumerate(history.history["val_loss"]), key=lambda x: x[1])
    best_mean_loss, best_val_mean_loss = best_sum_loss / height / width, best_val_sum_loss / height / width
    eval_mean_loss, eval_sum_loss = evaluation[0] / height / width, evaluation[0]
else:
    best_loss_epoch, best_mean_loss = min(enumerate(history.history["loss"]), key=lambda x: x[1])
    best_val_loss_epoch, best_val_mean_loss = min(enumerate(history.history["val_loss"]), key=lambda x: x[1])
    best_sum_loss, best_val_sum_loss = best_mean_loss * height * width, best_val_mean_loss * height * width
    eval_mean_loss, eval_sum_loss = evaluation[0], evaluation[0] * height * width
best_acc_epoch, best_acc = max(enumerate(history.history["acc"]), key=lambda x: x[1])
best_val_acc_epoch, best_val_acc = max(enumerate(history.history["val_acc"]), key=lambda x: x[1])
results = dict(best_loss=dict(mean=best_mean_loss, sum=best_sum_loss, epoch=best_loss_epoch + 1),
               best_acc=dict(acc=best_acc, epoch=best_acc_epoch + 1),
               best_val_loss=dict(mean=best_val_mean_loss, sum=best_val_sum_loss, epoch=best_val_loss_epoch + 1),
               best_val_acc=dict(acc=best_val_acc, epoch=best_val_acc_epoch + 1),
               eval_loss=dict(mean=eval_mean_loss, sum=eval_sum_loss), eval_acc=dict(acc=evaluation[1]),
               loss=history.history["loss"], acc=history.history["acc"],
               val_loss=history.history["val_loss"], val_acc=history.history["val_acc"])
with open(os.path.join(results_dir, "results.json"), "w") as file:
    json.dump(results, file, indent=4)
