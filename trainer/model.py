import itertools

from keras import backend as K, optimizers
from keras import layers
from keras import models

import tensorflow as tf
import numpy as np

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

np.random.seed(1)
tf.set_random_seed(1)


def model_fn(input_dim=3,
             labels_dim=1,
             hidden_units=(3, ),
             **args):
    """Create a Keras Sequential model with layers.

    Args:
    input_dim: (int) Input dimensions for input layer.
    labels_dim: (int) Label dimensions for input layer.
    hidden_units: [int] the layer sizes of the DNN (input layer first)
    learning_rate: (float) the learning rate for the optimizer.

    Returns:
        A Keras model.
    """

    # "set_learning_phase" to False to avoid:
    # AbortionError(code=StatusCode.INVALID_ARGUMENT during online prediction.
    K.set_learning_phase(0)

    learning_rate = float(args.get('learning_rate', 0.7))
    model = models.Sequential()
    # Input - Layer
    model.add(layers.Dense(3, input_dim=input_dim))
    # Hidden - Layer
    for units in hidden_units:
        model.add(layers.Dense(units=units, activation="sigmoid"))
    # Output- Layer
    model.add(layers.Dense(labels_dim, activation="sigmoid"))
    model.summary()
    print("Set {} as learning rate on model".format(learning_rate))
    compile_model(model, learning_rate)
    return model


def compile_model(model, learning_rate):
    model.compile(optimizer=optimizers.SGD(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def to_saved_model(model, export_path):
    """Convert the Keras HDF5 model into TensorFlow SavedModel."""

    builder = saved_model_builder.SavedModelBuilder(export_path)

    signature = predict_signature_def(
        inputs={'input': model.inputs[0]}, outputs={'income': model.outputs[0]})

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
            })
        builder.save()

    print("Model saved in GCS")


def generator_input(batch_size=2):
    def operation(a, b, c):
        return 1 if (a != b and (a or c)) else 0
    while True:
        x_input = np.array(list(map(list, itertools.product([0, 1], repeat=3))))
        y_label = np.array(list(map(lambda f: operation(f[0], f[1], f[2]), x_input)))
        idx_len = x_input.shape[0]
        for index in range((0 + batch_size), (idx_len + 1), batch_size):
            yield (x_input[index-batch_size: index],
                   y_label[index-batch_size: index])
