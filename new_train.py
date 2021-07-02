import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

def build(width, height, depth, classes):
		# initialize the model
		model = Sequential()
		inputShape = (height, width, depth)

		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)

		# first set of CONV => RELU => POOL layers
		model.add(Conv2D(20, (5, 5), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# second set of CONV => RELU => POOL layers
		model.add(Conv2D(50, (5, 5), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model


modelname = "final.model"
data = []
labels = []
path = 'train_images\\'
resize = 28

EPOCHS = 30
INIT_LR = 1e-3
BS = 75

for filename in os.listdir(path):
    if not filename.endswith('.xml'): continue
    fullname = os.path.join(path, filename)
    print(fullname)
    tree = ET.parse(fullname)

    root = tree.getroot()

    fileimg = root[1].text
    print(fileimg)
    image = cv2.imread(path + fileimg)
    
    
    objects = root.findall('object')
    for obj in objects:
    #    print (obj.tag, obj.attrib)
        
        name = obj[0].text
#        print(name)
        items = obj.findall('bndbox')
        xmin = int(items[0][0].text)
        ymin = int(items[0][1].text)
        xmax = int(items[0][2].text)
        ymax = int(items[0][3].text)
        print(xmin,ymin,xmax,ymax)
        patch = image[ymin:ymax,xmin:xmax,:]
        patch = cv2.resize(patch, (resize, resize))
#        x = np.array(patch).flatten()
        x = img_to_array(patch)
		
        data.append(x) 
        
        label = 0 
        if name == "masked":
            label=1
		
        elif name == "maskless":
            label=0
		


        labels.append(label)


data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
model = build(width=resize, height=resize, depth=3, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] saving network...")
model.save(modelname, save_format="h5")


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history['loss'], label="train_loss")
plt.plot(np.arange(0, N), H.history['val_loss'], label="val_loss")
plt.plot(np.arange(0, N), H.history['accuracy'], label="train_acc")
plt.plot(np.arange(0, N), H.history['val_accuracy'], label="val_acc")
plt.title("Training Loss and Accuracy on " + modelname)
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
