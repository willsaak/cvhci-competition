import os
import cv2
import numpy as np
from keras import callbacks


class TestPredictor(callbacks.Callback):
    """
    Save test predictions after every epoch.
    """
    def __init__(self, test_generator, path, save_prefix="", background_as_class=False):
        super(TestPredictor, self).__init__()
        self.test_generator = test_generator
        self.dst_path = path
        self.background_as_class = background_as_class
        self.save_prefix = save_prefix

    def on_epoch_end(self, epoch, logs=None):
        results = self.model.predict_generator(
            self.test_generator,
            steps=self.test_generator.samples / self.test_generator.batch_size,
            workers=0,  # highly important, otherwise image order can be shuffled in second iterations
            verbose=0
        )
        if self.background_as_class is True:
            # Cut the background channel
            segmentations = np.array(results[..., 1:] * 255., dtype=np.uint8)
        else:
            segmentations = np.array(results * 255., dtype=np.uint8)
        dst_dir = os.path.join(self.dst_path, "epoch_%03d" % (epoch + 1))
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
        if segmentations.shape[-1] == 1:
            for filename, segmentation in zip(self.test_generator.filenames, segmentations):
                cv2.imwrite(os.path.join(dst_dir, self.save_prefix + os.path.split(filename)[-1]), segmentation)
        else:
            # Save multi-class and thus multi-channel segmentations,
            # for example convert classes to different colors
            # Unnecessary as long as no loading of multi-class masks for training is implemented
            pass
        print('\nEpoch %05d: saving test results to %s' % (epoch + 1, dst_dir))