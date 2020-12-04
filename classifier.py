from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np

class Classifier:
    def __init__(self, ):
        self.model = None
        self.image_size = (0, 0)
        pass

    # training methods

    def pre_processing(self, path_dataset, batch_size, image_size, class_mode):
        self.image_size = image_size
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest"
        )
        train_generator = train_datagen.flow_from_directory(
            path_dataset + "train/",
            target_size=image_size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=True)

        validation_datagen = ImageDataGenerator(rescale=1. / 255)

        validation_generator = validation_datagen.flow_from_directory(
            path_dataset + "validation/",
            target_size=image_size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(
            path_dataset + "test/",
            target_size=image_size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=False)

        # print(train_generator.class_indices)
        return train_generator, validation_generator, test_generator

    def configure_model(self):
        model = Sequential()

        model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu',
                         input_shape=(self.image_size[0], self.image_size[1], 3)))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        model.add(Dropout(0.4))
        model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
        model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        model.add(Dropout(0.4))
        model.add(Conv2D(filters=256, kernel_size=4, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(16, activation="softmax"))

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self.model = model

    def model_print_summary(self):
        print(self.model.summary())

    def model_training(self, train_generator, validation_generator, epochs_number):
        hist = self.model.fit(train_generator, validation_data=validation_generator,
                              epochs=epochs_number, )

    def model_evaluate(self, generator):
        metrics = self.model.evaluate(generator)
        return metrics

    def model_save(self, model_name):
        self.model.save(model_name + ".h5")

    # predict methods
    def set_image_size(self, img_size):
        self.image_size = img_size

    def model_load(self, path_model):
        self.model = load_model(path_model)

    def model_predict_img(self, img):
        # get max class probability
        class_predicted = np.argmax(self.model.predict(img), axis=-1)
        # set labels classes
        #class_dictionary = {'abraham_grampa': 0, 'apu': 1, 'bart': 2, 'charles_montgomery_burns': 3, 'chief_wiggum': 4, 'comic_book_guy': 5, 'edna_krabappel': 6, 'homer': 7, 'krusty_the_clown': 8, 'lisa': 9, 'marge': 10, 'milhouse_van_houten': 11, 'moe_szyslak': 12, 'ned_flanders': 13, 'principal_skinner': 14, 'sideshow_bob': 15}
        class_dictionary = {'Abraham': 0, 'Apu': 1, 'Bart': 2, 'Montgomery Burns': 3, 'Chief Wiggum': 4, 'Comic Book Guy': 5, 'Edna Krabappel': 6, 'Homer': 7, 'Krusty the clown': 8, 'Lisa': 9, 'Marge': 10, 'Milhouse': 11, 'Moe Szyslak': 12, 'Ned Flanders': 13, 'Principal Skinner': 14, 'Sideshow Bob': 15}
        # invert class_dictionary
        inv_map = {v: k for k, v in class_dictionary.items()}
        #print("ID: {}, Label: {}".format(class_predicted[0], inv_map[class_predicted[0]]))
        # inv_map[class_predicted[0]] include label class predicted from model
        return inv_map[class_predicted[0]]

