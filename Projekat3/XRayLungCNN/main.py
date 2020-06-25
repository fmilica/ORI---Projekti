from pathlib import Path
import cv2
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Dropout, Input, Flatten, SeparableConv2D
import keras

#image width and height
img_rows, img_cols = 64, 64

epochs = 70
batch_size = 16
num_classes = 3

train_dir = Path("data/train")
val_dir = Path("data/val")

if __name__ == '__main__':

    #--------------Training data-------------
    normal_cases_dir = train_dir / 'NORMAL'
    bacteria_cases_dir = train_dir / 'BACTERIA'
    virus_cases_dir = train_dir / 'VIRUS'

    #Lists of all images
    normal_cases = normal_cases_dir.glob('*.jpeg')
    bacteria_cases = bacteria_cases_dir.glob('*.jpeg')
    virus_cases = virus_cases_dir.glob('*.jpeg')

    # List that are going to contain validation images data and the corresponding labels
    train_data = []
    train_labels = []

    # Normalize the pixel values and resizing all the images to 64x64
    #Normal cases
    for img in normal_cases:
        img = cv2.imread(str(img))
        img = cv2.resize(img, (img_rows, img_cols))
        img = img.astype(np.float32) / 255.
        label = to_categorical(0, num_classes=3)
        train_data.append(img)
        train_labels.append(label)
    # Bacteria cases
    for img in bacteria_cases:
        img = cv2.imread(str(img))
        img = cv2.resize(img, (img_rows, img_cols))
        img = img.astype(np.float32) / 255.
        label = to_categorical(1, num_classes=3)
        train_data.append(img)
        train_labels.append(label)
    # Virus cases
    for img in virus_cases:
        img = cv2.imread(str(img))
        img = cv2.resize(img, (img_rows, img_cols))
        img = img.astype(np.float32) / 255.
        label = to_categorical(2, num_classes=3)
        train_data.append(img)
        train_labels.append(label)

    # Convert the list into numpy arrays
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    # --------------Validation data-------------
    normal_cases_dir = val_dir / 'NORMAL'
    bacteria_cases_dir = val_dir / 'BACTERIA'
    virus_cases_dir = val_dir / 'VIRUS'

    # Get the list of all the images
    normal_cases = normal_cases_dir.glob('*.jpeg')
    bacteria_cases = bacteria_cases_dir.glob('*.jpeg')
    virus_cases = virus_cases_dir.glob('*.jpeg')

    # List that are going to contain validation images data and the corresponding labels
    val_data = []
    val_labels = []

    # Normalize the pixel values and resizing all the images to 64x64
    #Normal cases
    for img in normal_cases:
        img = cv2.imread(str(img))
        img = cv2.resize(img, (img_rows, img_cols))
        img = img.astype(np.float32) / 255.
        label = to_categorical(0, num_classes=3)
        val_data.append(img)
        val_labels.append(label)
    # Bacteria cases
    for img in bacteria_cases:
        img = cv2.imread(str(img))
        img = cv2.resize(img, (img_rows, img_cols))
        img = img.astype(np.float32) / 255.
        label = to_categorical(1, num_classes=3)
        val_data.append(img)
        val_labels.append(label)
    # Virus cases
    for img in virus_cases:
        img = cv2.imread(str(img))
        img = cv2.resize(img, (img_rows, img_cols))
        img = img.astype(np.float32) / 255.
        label = to_categorical(2, num_classes=3)
        val_data.append(img)
        val_labels.append(label)

    # Convert the list into numpy arrays
    val_data = np.array(val_data)
    val_labels = np.array(val_labels)

    #Data is processed, now we need to create the model

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(img_rows, img_cols, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), input_shape=(img_rows, img_cols, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), input_shape=(img_rows, img_cols, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), input_shape=(img_rows, img_cols, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(64, activation= "relu", kernel_regularizer=keras.regularizers.l2(0.001)))

    model.add(Dropout(0.5))

    model.add(Dense(3, kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(Activation("softmax"))

    model.summary()

    # Compile model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    # Train and evaluation
    model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(val_data, val_labels))
    # Show results
    score = model.evaluate(val_data, val_labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])