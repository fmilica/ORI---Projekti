from pathlib import Path
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, Activation, Dense, \
    Dropout, Flatten, LeakyReLU, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import to_categorical

import keras

#image width and height
img_rows, img_cols = 64, 64

epochs = 20
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
    normal_data = []
    normal_labels = []

    virus_data = []
    virus_labels = []

    bacteria_data = []
    bacteria_labels = []

    train_data = []
    train_labels = []

    test_data = []
    test_labels = []

    # Normalize the pixel values and resizing all the images to 64x64
    # sNormal cases
    num_normal = 0
    for img in normal_cases:
        num_normal += 1
        img = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_rows, img_cols))
        #img = img / 255.
        label = to_categorical(0, num_classes=3)
        normal_data.append(img)
        normal_data.append(label)
    normal_num = int(len(normal_data) * 0.8)
    train_data.extend(normal_data[:normal_num])
    train_labels.extend(normal_labels[:normal_num])
    # Bacteria cases
    num_bacteria = 0
    for img in bacteria_cases:
        num_bacteria += 1
        img = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_rows, img_cols))
        #img = img / 255.
        label = to_categorical(1, num_classes=3)
        bacteria_data.append(img)
        bacteria_labels.append(label)
    bacteria_num = int(len(bacteria_data) * 0.8)
    train_data.extend(bacteria_data[:bacteria_num])
    train_labels.extend(bacteria_labels[:bacteria_num])
    # Virus cases
    num_virus = 0
    for img in virus_cases:
        num_virus += 1
        img = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_rows, img_cols))
        #img = img / 255.
        label = to_categorical(2, num_classes=3)
        virus_data.append(img)
        virus_labels.append(label)
    virus_num = int(len(virus_data) * 0.8)
    train_data.extend(virus_data[:virus_num])
    train_labels.extend(virus_labels[:virus_num])

    # Convert the list into numpy arrays
    train_data = np.array(train_data)
    train_data = train_data.reshape([len(train_data), img_rows, img_cols, 1])
    train_labels = np.array(train_labels)

    test_data = np.array(test_data)
    test_data = test_data.reshape([len(test_data), img_rows, img_cols, 1])
    test_labels = np.array(test_labels)

    '''
    # Process data

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        validation_split=0.2
        )

    train_generator = train_datagen.flow_from_directory(
        'data/train',
        #labels="inferred",
        #label_mode="categorical",
        #class_names=None,
        class_mode='categorical',
        color_mode="grayscale",
        batch_size=batch_size,
        #image_size=(64, 64),
        target_size=(img_cols, img_rows),
        shuffle=True,
        seed=None,
        #validation_split=None,
        subset='training',
        interpolation="bilinear",
        follow_links=False,
    )

    #val_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = train_datagen.flow_from_directory(
        'data/train',
        #labels="inferred",
        #label_mode="categorical",
        #class_names=None,
        class_mode='categorical',
        color_mode="grayscale",
        batch_size=batch_size,
        #image_size=(64, 64),
        target_size=(img_rows, img_cols),
        shuffle=True,
        seed=None,
        #validation_split=None,
        subset='validation',
        interpolation="bilinear",
        follow_links=False
    )
    '''

    #Data is processed, now we need to create the model

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", input_shape=(img_rows, img_cols, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, padding='same', activation="relu", kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, padding='same', activation="relu", kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, padding='same', activation="relu", kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(64, padding='same', activation="relu", kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    #model.add(Dense(128, activation="relu"))
    #model.add(Dropout(0.7))

    #model.add(Dense(64, activation="relu"))
    #model.add(Dropout(0.5))

    model.add(Dense(3, kernel_regularizer=keras.regularizers.l2(0.001), activation="softmax"))

    model.summary()

    # Compile model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    # Train and evaluation
    model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.2,
              shuffle=True, verbose=2)
    #model.fit_generator(generator=train_generator, validation_data=validation_generator, epochs=epochs, verbose=1)
    #model.fit_generator(generator=train_generator, )
    # Show results
    #score = model.evaluate_generator(validation_generator)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])

    predictions = model.predict(test_data)
    predictions_max = np.array(np.argmax(predictions, axis=1))
    check = predictions_max == test_labels
    count = 0
    for c in check:
        if c:
            count+=1
    print(count)
    print("Finished evaluation")
    print("Result: ", count, "/", len(test_labels), " correct")
    print("Accuracy: ", count / len(test_labels) * 100, "%")
