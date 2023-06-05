import sys
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import os
from model import CNN_Encoder, RNN_Decoder, MY_EMBEDDING_DIM, UNIT_COUNT, WORD_DICT_SIZE
import datetime

# file parameters
feature_extraction_model = "xception"
model_dir = "trained_model_xception_yolo_boundingboxes-testing"

# change the current working directory to the parent directory of this code file # hadie
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # hadie
from yolo import image_path_to_yolo_bounding_boxes  # hadie

file = model_dir + "/max_length.txt"  # hadie
with open(file, 'r') as filetoread:  # hadie
    max_length = int(filetoread.readline())  # hadie

vocab_size = WORD_DICT_SIZE + 1
encoder = CNN_Encoder(MY_EMBEDDING_DIM)
decoder = RNN_Decoder(MY_EMBEDDING_DIM, UNIT_COUNT, vocab_size)
decoder.load_weights(model_dir + "/my_model")

import importlib  # hadie
mod = importlib.import_module("feature_extraction_model_" + feature_extraction_model)  # hadie
image_model = mod.image_model  # hadie
load_image = mod.load_image  # hadie
attention_features_shape = mod.attention_features_shape + 1  # hadie

new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

# loading the tokenizer
with open(model_dir + "/tokenizer.pickle", 'rb') as handle:
    tokenizer = pickle.load(handle)


def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()

features_shape = 2048
def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    yolo_features = image_path_to_yolo_bounding_boxes(image)  # hadie
    yolo_features = np.array(yolo_features.flatten())  # hadie
    yolo_features = np.pad(yolo_features, (0, features_shape - yolo_features.shape[0]), 'constant', constant_values=(0, 0)).astype(np.float32)  # hadie
    combined_features = np.vstack((img_tensor_val[0].numpy(), yolo_features)).astype(np.float32)  # hadie
    features = encoder(combined_features)  # hadie

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
          
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result),:]
    return result, attention_plot


image = sys.argv[1]
result, attention_plot = evaluate(image)
file_name = ('caption_sample_' + str(datetime.datetime.now()) + ".png").replace(":", "_").replace(" ", "__")
print ('Prediction Caption:`', ' '.join(result))
# plt.savefig(file_name)
print("The results have been written to " + file_name)
# plot_attention(image, result, attention_plot)
