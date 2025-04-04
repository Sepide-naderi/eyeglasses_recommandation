import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('face_shape_model.keras')

face_shapes = ['heart', 'oblong', 'oval', 'round', 'square']

def predict_face_shape(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))
    img_arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    prediction = model.predict(img_arr)
    predicted_shape = face_shapes[np.argmax(prediction)]

    return predicted_shape


img_path = 'oval1.jfif'
face_shape = predict_face_shape(img_path)
print(f"Predicted Face Shape: {face_shape}")
