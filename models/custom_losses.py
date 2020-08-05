from keras import backend


def image_binary_crossentropy(y_true, y_pred):
    square_wise_binary_crossentropy = backend.binary_crossentropy(y_true, y_pred)
    # backend.mean instead of backend.sum gives the same effect for training
    # as just using standard Keras binary_crossentropy
    return backend.sum(square_wise_binary_crossentropy, axis=(-1, -2, -3))


def image_categorical_crossentropy(y_true, y_pred):
    square_wise_categorical_crossentropy = backend.categorical_crossentropy(y_true, y_pred)
    # backend.mean instead of backend.sum gives the same effect for training
    # as just using standard Keras categorical_crossentropy
    return backend.sum(square_wise_categorical_crossentropy, axis=(-1, -2))