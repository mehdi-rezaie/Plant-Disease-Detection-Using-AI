import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Define the path to the folder containing the training images of diseased plants
training_folder = 'C:/Users/Mehdi OA/Desktop/PlantDiseaseDetection/training_folder/apple'

# Define the input image dimensions
input_shape = (256, 256, 3)

# Define the hyperparameters
batch_size = 32
epochs = 10

# Data augmentation for the training images
train_data_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Load the training images
train_data = train_data_gen.flow_from_directory(
    training_folder,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='binary'
)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, epochs=epochs)

# Save the trained model
model.save('C:/Users/Mehdi OA/Desktop/PlantDiseaseDetection/model/applemodel.h5')
