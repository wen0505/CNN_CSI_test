from keras.layers import Convolution2D, LSTM, Reshape
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

batch_size = 32
target_size = (64, 64)

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Reshaping the output of the previous layer to a 3D tensor
classifier.add(Reshape((1, -1)))

# Add a LSTM Layer
classifier.add(LSTM(units=128))

# Step 4 - Full connection
# classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('dataset2/train_set',
                                                 target_size=target_size,
                                                 batch_size=batch_size,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset2/test_set',
                                            target_size=target_size,
                                            batch_size=batch_size,
                                            class_mode='binary')

classifier.fit(training_set,
               epochs=25,
               steps_per_epoch=int(6000 // batch_size),
               validation_data=test_set,
               validation_steps=int(1000 // batch_size))
