import tensorflow as tf

attention_features_shape = 100


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.xception.preprocess_input(img)
    return img, image_path


image_model = tf.keras.applications.Xception(include_top=False,
                                                weights='imagenet')
