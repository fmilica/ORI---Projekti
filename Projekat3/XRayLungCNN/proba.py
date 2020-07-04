from pathlib import Path
import random
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Activation, Dropout, MaxPooling2D, Flatten, Dense, \
    LeakyReLU, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax
from tensorflow.keras.callbacks import TensorBoard

'''
Kako pokretati?
    1. Promeniti SAVE_MODEL_NAME shodno izmenama modela
    2. Pokrenuti proba.py
    3. Podici cmd u folderu XRayLungCNN
    4. Otvoriti venv/Scripts
    5. Pokrenuti fajl activate.bat (trebalo bi da vidite '(venv)' ispred putanje u komandnoj liniji)
    6. Ukucati komandu -> tensorboard --logdir=logs/
    7. Kada se komanda potpuno izvrsi dobicete adresu koju ubacite u browser
    8. And voila, pregled uspesnosti svih modela do sada pokrenutih
'''

'''
Smernice za poboljsanje:
    -> verovatno uracunati da skup nije skaliran dobro 
        (train ima najvise normal, test ima najvise bacteria)
'''

TRAIN_DIR = Path.cwd() / Path("data/train")
TEST_DIR = Path.cwd() / Path("data/chest-xray-dataset-test/test")
TEST_LABELS = Path.cwd() / Path("data/chest-xray-dataset-test/chest_xray_test_dataset.csv")
CATEGORIES = ["NORMAL", "BACTERIA", "VIRUS"]
NUM_CLASSES = 3
IMG_ROWS, IMG_COLS = 64, 64
BATCH_SIZE = 32
EPOCHS = 20
#64x4-OneDropout-Batch-16
# mozda je cak pet slojeva bolje
# X-Ray-CNN-64x5-TwoDropout-Batch-32-l1-0x001
# sad cemo da dodamo dropout posle svakog
#BEST_MODEL_NAME = "X-Ray-CNN-64x4-OneDropout-Batch-32-l1-0x001"
SAVE_MODEL_NAME = "X-Ray-CNN-64x5-OneDropout-Batch-32-l1-0x001"
MODEL_NAME = SAVE_MODEL_NAME + "-{}".format(int(time.time()))
tensor_board = TensorBoard(log_dir="logs/{}".format(MODEL_NAME))


# CREATING TEST DATA
def create_test_labels():
    # read labeled test data .csv file
    test_labels_df = pd.read_csv(TEST_LABELS)
    test_labels_df = test_labels_df.fillna('NORMAL')
    labels_dict = pd.Series(test_labels_df.Label_1_Virus_category.values,
                            index=test_labels_df.X_ray_image_name).to_dict()
    labels_dict = dict((k, v.upper()) for k, v in labels_dict.items())
    return labels_dict


def create_test_data():
    testing_data = []
    labels_data_dict = create_test_labels()
    for img in tqdm(TEST_DIR.glob('*jpeg')):
        img_name = img.parts[-1]
        try:
            category = labels_data_dict[img_name]
        except KeyError:
            continue
        class_num = CATEGORIES.index(category)
        img_array = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, (IMG_ROWS, IMG_COLS))
        testing_data.append([new_img_array, class_num])
    # shuffle the data
    random.shuffle(testing_data)
    # create np arrays
    test_x = []
    test_y = []

    for features, label in testing_data:
        test_x.append(features)
        test_y.append(label)

    # reshape np array
    test_x = np.array(test_x).reshape(-1, IMG_ROWS, IMG_COLS, 1)
    test_y = np.array(test_y)
    test_y = to_categorical(test_y, 3)
    return test_x, test_y


def pickle_dump_test_data(test_x, test_y):
    # write prepared test data
    pickle_out = open("data/processed/test_data.pickle", 'w+b')
    pickle.dump(test_x, pickle_out)
    pickle_out.close()
    # write prepared training labels
    pickle_out = open("data/processed/test_labels.pickle", 'w+b')
    pickle.dump(test_y, pickle_out)
    pickle_out.close()


def pickle_load_test_set():
    # read prepared test data
    pickle_in = open("data/processed/test_data.pickle", "rb")
    test_x = pickle.load(pickle_in)
    # normalise the data
    test_x = test_x / 255.0
    # read prepared training labels
    pickle_in = open("data/processed/test_labels.pickle", "rb")
    test_y = pickle.load(pickle_in)
    return test_x, test_y


# CREATING TRAIN DATA
def create_train_data():
    training_data = []
    for category in CATEGORIES:
        category_path = TRAIN_DIR / category
        class_num = CATEGORIES.index(category)
        for img in tqdm(category_path.glob('*.jpeg')):
            img_array = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
            new_img_array = cv2.resize(img_array, (IMG_ROWS, IMG_COLS))
            training_data.append([new_img_array, class_num])
    # shuffle the data
    random.shuffle(training_data)
    # create np arrays
    train_x = []
    train_y = []

    for features, label in training_data:
        train_x.append(features)
        train_y.append(label)

    # reshape np array
    train_x = np.array(train_x).reshape(-1, IMG_ROWS, IMG_COLS, 1)
    train_y = np.array(train_y)
    train_y = to_categorical(train_y, 3)
    return train_x, train_y


def pickle_dump_train_set(train_x, train_y):
    # write prepared training data
    pickle_out = open("data/processed/train_data.pickle", 'w+b')
    pickle.dump(train_x, pickle_out)
    pickle_out.close()
    # write prepared training labels
    pickle_out = open("data/processed/train_labels.pickle", 'w+b')
    pickle.dump(train_y, pickle_out)
    pickle_out.close()


def pickle_load_train_set():
    # read prepared training data
    pickle_in = open("data/processed/train_data.pickle", "rb")
    train_x = pickle.load(pickle_in)
    # normalise the data
    train_x = train_x / 255.0
    # read prepared training labels
    pickle_in = open("data/processed/train_labels.pickle", "rb")
    train_y = pickle.load(pickle_in)

    return train_x, train_y


# CREATE MODEL
def create_model():
    model = Sequential()

    model.add(Conv2D(64, padding="same", kernel_size=(3, 3), input_shape=(IMG_ROWS, IMG_COLS, 1)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, padding='same', kernel_size=(3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, padding='same', kernel_size=(3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, padding='same', kernel_size=(3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.3))

    model.add(Conv2D(64, padding='same', kernel_size=(3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(3, kernel_regularizer=regularizers.l1(0.001)))
    model.add(Activation("softmax"))

    model.summary()
    return model


def compile_fit_model(model, train_x, train_y):
    # Compile model
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy'])

    # Train and evaluation
    model.fit(train_x, train_y,
              batch_size=BATCH_SIZE, epochs=EPOCHS,
              validation_split=0.2, verbose=1,
              callbacks=[tensor_board])

    # Save the trained model
    model.save(SAVE_MODEL_NAME + '.model')


# PREDICTIONS
def predict_test(test_x, test_y):
    # load the trained model
    loaded_model = load_model(SAVE_MODEL_NAME + '.model')

    # create predictions for test data
    predictions = loaded_model.predict([test_x])

    # compare results
    result = []
    for prediction, truth in zip(predictions, test_y):
        # get category index of prediction
        prediction_category = max(np.where(prediction == max(prediction))[0])
        # get category index of label
        truth_category = max(np.where(truth == 1)[0])
        result.append(1 == truth_category)
    print(result)


if __name__ == '__main__':
    # preprocess training data to np array and write to file
    #train_data, train_labels = create_train_data()
    #pickle_dump_train_set(train_data, train_labels)

    # preprocess test data to np array and write to file
    #test_data, test_labels = create_test_data()
    #pickle_dump_test_data(test_data, test_labels)

    # read preprocessed training and test data
    train_data, train_labels = pickle_load_train_set()
    #test_data, test_labels = pickle_load_test_set()

    # create CNN model
    cnn_model = create_model()
    compile_fit_model(cnn_model, train_data, train_labels)

    # load the trained model
    #saved_model = load_model(SAVE_MODEL_NAME + '.model')

    # evaluacija
    #test_eval = saved_model.evaluate(test_data, test_labels, verbose=1, batch_size=32)

    # BEST MODEL
    #best_model = load_model(BEST_MODEL_NAME + '.model')
    #test_eval = best_model.evaluate(test_data, test_labels, verbose=1, batch_size=32)

    # test CNN model on not-seen-before data
    # OVDE NESTO NE RADI ALI SADA NE VIDIM TACNO STA
    #predict_test(test_data, test_labels)
