import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import os


num_classes = 5
image_rows, img_cols = 48, 48
batch_size = 32
train_data_dir = r'D:\Desktop\ml-keras\train'
validation_data_dir = r'D:\Desktop\ml-keras\validation'

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=30, shear_range=0.3, zoom_range=0.3, width_shift_range=0.4, height_shift_range=0.4, horizontal_flip=True, vertical_flip=True
                                   )

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir, color, color_mode='grayscale', target_size=(
    image_rows, img_cols), batch_size=batch_size, class_mode='categorical', shuffle=True)

validation_generator = validation_datagen.flow_from_directory(validation_data_dir, color, color_mode='grayscale', target_size=(
    image_rows, img_cols), batch_size=batch_size, class_mode='categorical', shuffle=True)


model = Sequential()

# ---------------------------------------------------------------------------- #
#                                    block 1                                   #
# ---------------------------------------------------------------------------- #


model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer=(
    'he_normal'), input_shape=(image_rows, img_cols, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer=(
    'he_normal'), input_shape=(image_rows, img_cols, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# ---------------------------------------------------------------------------- #
#                                    block 2                                   #
# ---------------------------------------------------------------------------- #

model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=(
    'he_normal'), input_shape=(image_rows, img_cols, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=(
    'he_normal'), input_shape=(image_rows, img_cols, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# ---------------------------------------------------------------------------- #
#                                    block 3                                   #
# ---------------------------------------------------------------------------- #


model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer=(
    'he_normal'), input_shape=(image_rows, img_cols, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer=(
    'he_normal'), input_shape=(image_rows, img_cols, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# ---------------------------------------------------------------------------- #
#                                    block 4                                   #
# ---------------------------------------------------------------------------- #


model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=(
    'he_normal'), input_shape=(image_rows, img_cols, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=(
    'he_normal'), input_shape=(image_rows, img_cols, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


# ---------------------------------------------------------------------------- #
#                                    block 5                                   #
# ---------------------------------------------------------------------------- #

model.add(Flatten())
model.add(Dense(64, kernel_initializer=('he_normal')))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


# ---------------------------------------------------------------------------- #
#                                    block 6                                   #
# ---------------------------------------------------------------------------- #
