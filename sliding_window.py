import cv2
import matplotlib.pyplot as plt

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np


modelname = "final.model"
img_file = "C:\\t4zb\\3.sinif\\image_prossesing\\odev_new\\Final Test\\Images\\"
resize = 280


# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(modelname)

# load the image
image = cv2.imread(img_file+"maksssksksss1.png")
image = cv2.resize(image,(resize,resize))
temp = image # your image path
stepSizex = 35
stepSizey = 25
(w_width, w_height) = (50, 50) # window size
for x in range(0, image.shape[1] - w_width , stepSizex+20):
    for y in range(0, image.shape[0] - w_height, stepSizey+20):
        window = image[x:x + w_width, y:y + w_height, :]

        tempIMG = cv2.resize(window,(28,28))
        # pre-process the image for classification
        tempIMG = tempIMG.astype("float") / 255.0
        tempIMG = img_to_array(tempIMG)
        tempIMG = np.expand_dims(tempIMG, axis=0)
        
        # classify the input image
        (negative, pozitive) = model.predict(tempIMG)[0]

        # build the label
        label = "maskeli" if pozitive > negative else "maskesiz"
        kontrol = label
        proba = pozitive if pozitive > negative else negative
        label = "{}: {:.2f}%".format(label, proba * 100)
        oran = proba * 100

        if kontrol == "maskeli" and oran >70 and oran < 90:
            print("result : ",str(label))
        cv2.rectangle(image, (x, y), (x + w_width, y + w_height), (0, 255, 255), 1)
        plt.imshow(np.array(image))
        plt.pause(0.05)

plt.title('Sliding Window')
plt.show()
