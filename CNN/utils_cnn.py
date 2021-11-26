import tensorflow as tf
import numpy as np
import pathlib

AUTOTUNE = tf.data.AUTOTUNE

data_dir = pathlib.Path("../Dataset/")

image_count = len(list(data_dir.glob('*/*')))

batch_size = 32
img_height = 180
img_width = 180

train_data = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.25,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_data = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.25,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_data.class_names
print(class_names)

# Ubah gambar jadi hitam putih
normalization_layer = tf.keras.layers.Rescaling(1./255)
print("==========================================================")
print(normalization_layer)
print("==========================================================")
normalized_ds = train_data.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
test_data = test_data.cache().prefetch(buffer_size=AUTOTUNE)

