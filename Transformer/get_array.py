import numpy as np
import numpy as np
import tensorflow as tf
import glob

normal = glob.glob("../Dataset/normal/*")
glaucoma = glob.glob("../Dataset/glaucoma/*")

data = []
labels = []

for i in normal:   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
    target_size= (32,32))
    image=np.array(image)
    data.append(image)
    labels.append(0)

for i in glaucoma:   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
    target_size= (32,32))
    image=np.array(image)
    data.append(image)
    labels.append(1)

data = np.array(data)
labels = np.array(labels)

np.save('../Array/arr_transformer/data.txt', data)
np.save('../Array/arr_transformer/labels.txt', labels)