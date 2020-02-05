import os
import csv
import keras
import numpy as np
import sklearn
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, Lambda


def generator(samples, batch_size=32):
    """Generates batches of training features and labels.

    Args
        samples: driving log file records
    """
    # Fetch corresponding images, and build a generator
    num_samples = len(samples)
    while 1:
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                f = batch_sample[-1]
                original_name = batch_sample[0]
                name = f.replace("driving_log.csv", "") + "IMG/" + original_name.split("IMG/")[-1]
                center_image = mpimg.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                # Flip the image to augment training data
                images.append(np.fliplr(center_image))
                angles.append(-center_angle)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def vgg_model():
    """Builds a VGG model.
    """
    model = Sequential()
    # Normalize
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    # Crop
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation=None))

    model.compile(loss='mean_squared_error', optimizer="adam")

    return model


def nvidia_model():
    """Builds a nvidia model.
    """
    model = Sequential()

    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))

    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer="adam")

    return model


def train_model(model,
                train_generator,
                validation_generator,
                steps_per_epoch,
                validation_steps,
                model_dir,
                epochs=10):
    """Trains a model.
    """
    stopper = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1)
    checkpoint = keras.callbacks.callbacks.ModelCheckpoint(
        filepath=os.path.join("./", model_dir, "model_checkpoint.{epoch:02d}-{val_loss:.2f}.h5"))

    model_history = model.fit_generator(train_generator,
                                        steps_per_epoch=steps_per_epoch,
                                        validation_data=validation_generator,
                                        validation_steps=validation_steps,
                                        epochs=epochs,
                                        callbacks=[stopper, checkpoint])
    return model, model_history


def main(data_dirs, model_dir, model="nvidia", epochs=10):
    # Read in driving log files
    samples = []
    for f in data_dirs:
        with open(f) as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for line in reader:
                line.append(f)
                samples.append(line)

    # Create training and validation generators
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_generator = generator(train_samples)
    validation_generator = generator(validation_samples)

    # Build model
    if model == "vgg":
        model = vgg_model()
    elif model == "nvidia":
        model = nvidia_model()

    # Train model
    try:
        os.mkdir(os.path.join("./", model_dir))
    except:
        pass

    steps_per_epoch = np.ceil(len(train_samples) / 32)
    validation_steps = np.ceil(len(validation_samples) / 32)

    model, model_history = train_model(model,
                                       train_generator=train_generator,
                                       validation_generator=validation_generator,
                                       steps_per_epoch=steps_per_epoch,
                                       validation_steps=validation_steps,
                                       epochs=epochs,
                                       model_dir=model_dir)

    # Save model
    model.save(os.path.join("./", model_dir, "model.h5"))

    # Save history plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(model_history.history['loss'])
    ax.plot(model_history.history['val_loss'])
    ax.set_title('model mean squared error loss')
    ax.set_ylabel('mean squared error loss')
    ax.set_xlabel('epoch')
    ax.legend(['training set', 'validation set'], loc='upper right')
    fig.savefig(os.path.join(model_dir, "learning_graph.png"))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", required=False, action="append")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--epochs", required=False, type=int)
    ap.add_argument("--model", required=False, type=str)
    args = vars(ap.parse_args())
    data_dirs = args["data_dirs"]
    model_dir = args["model_dir"]
    epochs = args["epochs"]
    model = args["model"]

    model_dir = model + "_" + model_dir

    print(data_dirs)
    print(args)
    print("   ")
    if data_dirs is None:
        data_dirs = ["../beta_simulator_mac/data/track1_run2/driving_log.csv",
                     "../beta_simulator_mac/data/track1_reverse_run1/driving_log.csv"]
        print("Using default data")
    else:
        print("Using the following data: ", data_dirs)

    main(data_dirs, model_dir, model, epochs)
