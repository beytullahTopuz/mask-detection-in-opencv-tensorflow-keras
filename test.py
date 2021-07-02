# USAGE
# python test_network.py --model santa_not_santa.model --image images/examples/santa_01.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2


modelname = "final.model"
img_file = "C:\\t4zb\\3.sinif\\image_prossesing\\odev_new\\Final Test\\Images\\"
resize = 28


# load the image
image = cv2.imread(img_file + "maksssksksss0.png")

bbox = cv2.selectROI(image,False)

image = image[int(bbox[1]):int(bbox[1]+bbox[3]),int(bbox[0]):int(bbox[0]+bbox[2]),:]

selected = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (resize, resize))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(modelname)

# classify the input image
(negative, pozitive) = model.predict(image)[0]

# build the label
label = "maskeli" if pozitive > negative else "maskesiz"
proba = pozitive if pozitive > negative else negative
label = "{}: {:.2f}%".format(label, proba * 100)

# draw the label on the image
output = imutils.resize(selected, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()