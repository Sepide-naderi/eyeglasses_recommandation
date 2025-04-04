from keras import Sequential
from keras import layers
import tensorflow as tf

train_dataset = 'Faceshape_dataset'

face_shapes = ['heart', 'oval', 'oblong', 'square', 'round']

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(
    train_dataset,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical')

model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, epochs=20)

model.save('face_shape_model.keras')
