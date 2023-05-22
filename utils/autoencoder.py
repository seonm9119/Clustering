import keras
from keras import layers


def AE():
    encoding_dim = 32

    input_img = keras.Input(shape=(784,))
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    decoded = layers.Dense(784, activation='sigmoid')(encoded)
    autoencoder = keras.Model(input_img, decoded)
    

    encoder = keras.Model(input_img, encoded)
    encoded_input = keras.Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

    return autoencoder, encoder, decoder


def sparsity_AE():
    from keras import regularizers

    encoding_dim = 32
    
    input_img = keras.Input(shape=(784,))
    encoded = layers.Dense(encoding_dim, activation='relu',
                           activity_regularizer=regularizers.l1(10e-5))(input_img)
    decoded = layers.Dense(784, activation='sigmoid')(encoded)
    autoencoder = keras.Model(input_img, decoded)


    return autoencoder


def SAE():
    
    encoding_dim = 32
    input_img = keras.Input(shape=(784,))
    encoded = layers.Dense(128, activation='relu')(input_img)
    encoded = layers.Dense(64, activation='relu')(encoded)
    encoded = layers.Dense(32, activation='relu')(encoded)

    
    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dense(128, activation='relu')(decoded)
    decoded = layers.Dense(784, activation='sigmoid')(decoded)
    
    autoencoder = keras.Model(input_img, decoded)

    
    return autoencoder
    

def CAE():
    
    input_img = keras.Input(shape=(28, 28, 1))

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    

    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded)
    
    return autoencoder
    
    
