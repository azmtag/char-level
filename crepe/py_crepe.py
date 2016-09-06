from keras.models import Model
from keras.optimizers import SGD,Adam
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D


def model(filter_kernels, dense_outputs, maxlen, vocab_size, nb_filter, mode='1mse', cat_output=1, optimizer='adam'):
    print("Constructing model with mode %s, optimizer %s" % (mode, optimizer))
    #Define what the input shape looks like
    inputs = Input(shape=(maxlen, vocab_size), name='input', dtype='float32')

    #All the convolutional layers...
    conv = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[0],
                         border_mode='valid', activation='relu',
                         input_shape=(maxlen, vocab_size))(inputs)
    conv = MaxPooling1D(pool_length=3)(conv)

    conv1 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[1],
                          border_mode='valid', activation='relu')(conv)
    conv1 = MaxPooling1D(pool_length=3)(conv1)

    conv2 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[2],
                          border_mode='valid', activation='relu')(conv1)

    conv3 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[3],
                          border_mode='valid', activation='relu')(conv2)

    conv4 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[4],
                          border_mode='valid', activation='relu')(conv3)

    conv5 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[5],
                          border_mode='valid', activation='relu')(conv4)
    conv5 = MaxPooling1D(pool_length=3)(conv5)
    conv5 = Flatten()(conv5)

    #Two dense layers with dropout of .5
    z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(conv5))
    z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(z))

    #Output dense layer with softmax activation
    # pred = Dense(cat_output, activation='softmax', name='output')(z)

    ## now depending on the mode of input data we finish with different output layers
    if mode == '1mse':
      # Output dense layer with linear activation for predicting a continuous value
      pred = Dense(1, name='output')(z)
      model = Model(input=inputs, output=pred)
      model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    elif mode == 'cat':
      # Output dense layer with softmax activation for classification to cat_output classes
      pred = Dense(cat_output, activation='softmax', name='output')(z)
      model = Model(input=inputs, output=pred)
      model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    else:
      # Default: output dense layer with boolean for binary classification
      pred = Dense(1, name='output')(z)
      model = Model(input=inputs, output=pred)
      model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
