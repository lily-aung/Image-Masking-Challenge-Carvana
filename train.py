import threading
import cv2
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,TensorBoard
from sklearn.model_selection import train_test_split
from model import get_dilated_unet
import datetime

import config
WIDTH = config.input_size
HEIGHT = config.input_size
BATCH_SIZE = config.batch_size


class ThreadSafeIterator:

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*args, **kwargs):
        return ThreadSafeIterator(f(*args, **kwargs))

    return g

#**************************************************************************************************************#
# Data Augmentation
def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask
def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask
	
@threadsafe_generator
def train_generator(df):
    while True:
        shuffle_indices = np.arange(len(df))
        shuffle_indices = np.random.permutation(shuffle_indices)
        
        for start in range(0, len(df), BATCH_SIZE):
            x_batch = []
            y_batch = []
            
            end = min(start + BATCH_SIZE, len(df))
            ids_train_batch = df.iloc[shuffle_indices[start:end]]
            
            for _id in ids_train_batch.values:
                img = cv2.imread('input/train_hq/{}.jpg'.format(_id))
                img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                
                mask = cv2.imread('input/train_masks/{}_mask.png'.format(_id), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                mask = np.expand_dims(mask, axis=-1)
                assert mask.ndim == 3
				# Data Augmentation
				img = randomHueSaturationValue(img,
                                               hue_shift_limit=(-50, 50),
                                               sat_shift_limit=(-5, 5),
                                               val_shift_limit=(-15, 15))
                img, mask = randomShiftScaleRotate(img, mask,
                                                   shift_limit=(-0.0625, 0.0625),
                                                   scale_limit=(-0.1, 0.1),
                                                   rotate_limit=(-0, 0))
                #img, mask = randomHorizontalFlip(img, mask)
				
                if np.random.random() < 0.5:
                    img, mask = img[:, ::-1, :], mask[..., ::-1, :]  # random horizontal flip
                
                x_batch.append(img)
                y_batch.append(mask)
            
            x_batch = np.array(x_batch, np.float32) / 255.
            y_batch = np.array(y_batch, np.float32) / 255.
            
            yield x_batch, y_batch


@threadsafe_generator
def valid_generator(df):
    while True:
        for start in range(0, len(df), BATCH_SIZE):
            x_batch = []
            y_batch = []

            end = min(start + BATCH_SIZE, len(df))
            ids_train_batch = df.iloc[start:end]

            for _id in ids_train_batch.values:
                img = cv2.imread('input/train_hq/{}.jpg'.format(_id))
                img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

                mask = cv2.imread('input/train_masks/{}_mask.png'.format(_id),
                                  cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                mask = np.expand_dims(mask, axis=-1)
                assert mask.ndim == 3
                
                x_batch.append(img)
                y_batch.append(mask)

            x_batch = np.array(x_batch, np.float32) / 255.
            y_batch = np.array(y_batch, np.float32) / 255.

            yield x_batch, y_batch


if __name__ == '__main__':

    df_train = pd.read_csv('input/train_masks.csv')
    ids_train = df_train['img'].map(lambda s: s.split('.')[0])

    #ids_train, ids_valid = train_test_split(ids_train, test_size=0.1)
    print(datetime.datetime.now(), 'Perform stratified sampling into train-validation-test datasets (64%-16%-20%)...')
    ids_train_split, ids_test_split = train_test_split(ids_train, test_size=0.2, random_state=42)
    ids_train_split, ids_valid_split = train_test_split(ids_train_split, test_size=0.2, random_state=42)
    model = get_dilated_unet(
        input_shape=(1024, 1024, 3),
        mode='cascade',
        filters=32,
        n_class=1
    )

    callbacks = [EarlyStopping(monitor='val_dice_coef',
                               patience=10,
                               verbose=1,
                               min_delta=1e-4,
                               mode='max'),
                 ReduceLROnPlateau(monitor='val_dice_coef',
                                   factor=0.2,
                                   patience=5,
                                   verbose=1,
                                   epsilon=1e-4,
                                   mode='max'),
                 ModelCheckpoint(monitor='val_dice_coef',
                                 filepath='weights/best_weights.hdf5',
                                 save_best_only=True,
                                 mode='max'),
				 TensorBoard(log_dir='logs')]

    model.fit_generator(generator=train_generator(ids_train_split),
                        steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(BATCH_SIZE)),
                        epochs=100,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=valid_generator(ids_valid_split),
                        validation_steps=np.ceil(float(len(ids_valid_split)) / float(BATCH_SIZE)))
