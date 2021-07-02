
# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2
import os
import xml.etree.ElementTree as ET

modelname = "final.model"
img_file = "C:\\t4zb\\3.sinif\\image_prossesing\\odev_new\\Final Test\\Images\\"
xml_file = "C:\\t4zb\\3.sinif\\image_prossesing\\odev_new\\Final Test\\annotations\\"

resize = 28

print("[INFO] loading network...")
model = load_model(modelname)
for filename in os.listdir(xml_file):
    if not filename.endswith('.xml'): continue
    fullname = os.path.join(xml_file, filename)
    print(fullname)
    tree = ET.parse(fullname)

    root = tree.getroot()

    fileimg = root[1].text
    print(fileimg)



    objects = root.findall('object')
    for obj in objects:
        #print (obj.tag, obj.attrib)
        image = cv2.imread(img_file + fileimg)
        #image = cv2.resize(image, (resize, resize))
        
        # print("Shape : "+str(image.shape))

        items = obj.findall('bndbox')
        xmin = int(items[0][0].text)
        ymin = int(items[0][1].text)
        xmax = int(items[0][2].text)
        ymax = int(items[0][3].text)
        #print(xmin,ymin,xmax,ymax)
        #image = image[ymax:ymin,xmax:xmin]
        image = image[ymin:ymax,xmin:xmax]
        #selected = image.copy()
        
        print("çalişti")
        try:
            image = cv2.resize(image,(28,28))
        except Exception as e:
            print(str(e))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # classify the input image
        (negative, pozitive) = model.predict(image)[0]

        # build the label
        label = "maskeli" if pozitive > negative else "maskesiz"
        # print("result",label)
        proba = pozitive if pozitive > negative else negative
        #label = "{}: {:.2f}%".format(label, proba * 100)

        if label == "maskeli":
            print("["+fileimg+"] :"+"maskeli")
        if label == "maskesiz":
            print("["+fileimg+"] :"+"maskesiz")

