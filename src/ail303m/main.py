import os

import keras
import numpy as np
import pandas as pd
from keras import layers, ops

# alcohol, residual sugar, chlorides, total sulful dioxide, fixed acidity, type
NUM_FEATURES = 6
DATA_PATH = "data/"


def main():
    inputs = keras.Input(shape=(NUM_FEATURES,))
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="wine")
    model.summary()

    # keras.utils.plot_model(model, "model_info.png", show_shapes=True)
    # 
    # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    data = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255
    
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.RMSprop(),
        metrics=["accuracy"],
    )
    
    history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)
    
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])




if __name__ == "__main__":
    main()
