import cv2
import numpy as np
import random


#Maskeli insanlar için yapılan işlemler

imageNumberList = list() #resim indisi olanların dizisi
numbers = list() #resim indisi olmayanların dizisi

#for döngüsü ile var olan resimleri teker teker gez
for i in range(0,436): 

    #resmin konumunu al.
    temp = 'Train/Masked/'+str(i)+"-with-mask.jpg";

    #konumu alınan resmi oku.
    img = cv2.imread(temp)
    
    try:
        #gürültülü resmi göster
        #cv2.imshow("goruntu " + str(i),img)
        
        #gürültü ve görüntü bozuklukları giderilir
        bblur = cv2.bilateralFilter(img,6,40,40)
       
        #görüntüyü göster
        #cv2.imshow("goruntu2 " + str(i),bblur)

        #göriltü ve görüntü bozuklukları giderilen resmi kaydet.
        cv2.imwrite("ImageProcessingTrain/Masked/"+str(i)+"-with-mask.jpg",bblur)

        #indisi resim ile dolu olanları liste içerisine alır
        imageNumberList.append(i) #image numbers

        #işlemi gerçekleşen resmi ekranda yazdır.
        print(str(i)+"-with-mask.jpg")
        
    except:

        #resim olmayan indisleri tutar. Daha sonra sentetik veri çoğaltma ile
        #boş olan indislere resim ataması yapılır
        numbers.append(i) #empty image number

        #dolu olmayan indisleri ekrana yazar
        print("boş indis: ", i)




#sentetik veri çoğaltma yöntemi
#i değeri o andaki olmayan resim sayısıdır.
for i in numbers: 

    #var olan indis resimlerinden rastgele seçim yapılır.
    index = random.randint(0,len(imageNumberList)-1)

    #var olan resim indexi ekrana yazdırılır.
    print("random sayı değeri: ",imageNumberList[index])

    #gürültü ve görüntü bozuklukları olmayan resmin konumunu alır.
    temp = 'ImageProcessingTrain/Masked/'+str(imageNumberList[index])+'-with-mask.jpg';

    #konumu alınan resmi okur.
    syntheticImg = cv2.imread(temp)
    
    #resim satır ve sutun değerleri bulunur.
    rows, cols = syntheticImg.shape[:2]

    #klasörde bulunan resimlerin yarısı ile döndürme işlemi gerçekleştirilir.
    if i <= 267:
        
        #döndürme işlemi için -45 ile 45 derece arasında random sayı belirlenir.
        angle = random.randint(-45,45)
        rotation_matrix = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        img_rotation = cv2.warpAffine(syntheticImg,rotation_matrix,(cols,rows))

        #döndürülen resim ekranda gösterilir.
        #cv2.imshow("dondurme" + str(i),img_rotation)

        #döndürülen resim tekrar aynı klasöre kayıt edilir.
        cv2.imwrite("ImageProcessingTrain/Masked/"+str(i)+"-with-mask.jpg",img_rotation)
        
       
    else:
        #klasörde kalan diğer resimler simetri işlemi yapılır
        flipped = cv2.flip(syntheticImg,1) #horizontal image

        #oluşan resim ekranda gösterilir.
        #cv2.imshow("simetri" + str(i),flipped)

        #simetrisi alınan resim aynı klasöre yazdılır.
        cv2.imwrite("ImageProcessingTrain/Masked/"+str(i)+"-with-mask.jpg",flipped)
        

#maskeli insanların fotoları simetri ve döndürülme işlemi gerçekleştirildi.




#maskesiz insanlar için yapılan işlemler

imageNumberList = [] #resim indisi olanların dizisi
numbers = [] #resim indisi olmayanların dizisi

#tüm resimler teker teker gezilir
for i in range(0,243): 

    #resmin konumu alınır
    temp = 'Train/Maskless/'+str(i)+".jpg";

    #konumu alınan resimi okur.
    img = cv2.imread(temp)
    
    try:
        #resim ekranda gösterilir
        #cv2.imshow("goruntu" + str(i),img)
        
        #gürültü ve görüntü bozukluklar giderilir
        bblur = cv2.bilateralFilter(img,6,40,40)
       
        #gürültü ve görüntü bozuklukları giderilen resim ekranda gösterilir
        #cv2.imshow("goruntu2" + str(i),bblur)

        #son olarak resim başka bir klasöre kayıt edilir.
        cv2.imwrite("ImageProcessingTrain/Maskless/"+str(i)+".jpg",bblur)

        #dolu olan index diziye eklenir.
        imageNumberList.append(i) #image numbers

        #kayıt edilen resim ekrana yazdılır
        print(str(i)+".jpg")
        
    except:

        #resim olmayan indisleri tutar.
        numbers.append(i) #empty image number
        print(i)



#sentetik veri çoğaltma işlemi
        
#i değeri o andaki olmayan foto sayısı
for i in numbers: 

    #var olan sayılardan random olarak index seçer
    index = random.randint(0,len(imageNumberList)-1)

    #random sayı değerini ekrana yazar
    print("random sayı değeri: ",imageNumberList[index]) 

    #kayıt edilecek resim konumunu verir.
    temp = 'ImageProcessingTrain/Maskless/'+str(imageNumberList[index])+'.jpg';

    #resmin okuması gerçekleştirilir.
    syntheticImg = cv2.imread(temp)

    #resmin satır ve sutun sayıları alınır.
    rows, cols = syntheticImg.shape[:2]

    #klasörün yarısı döndürme yarısı ise simetri olarak ayrılır.
    if i <= 121:
        
        #döndürme işlemi
        angle = random.randint(-45,45)
        rotation_matrix = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        img_rotation = cv2.warpAffine(syntheticImg,rotation_matrix,(cols,rows))

        #döndürülmüş resim ekranda gösterilir.
        #cv2.imshow("dondurme" + str(i),img_rotation)

        #ilgili klasöre kayıt edilir.
        cv2.imwrite("ImageProcessingTrain/Maskless/"+str(i)+".jpg",img_rotation)
        
       
    else:
        #simetri işlemi
        flipped = cv2.flip(syntheticImg,1) #horizontal image

        #simerisi alınan resmi ekranda gösterilir.
        #cv2.imshow("simetri" + str(i),flipped)

        #resim klasöre kayıt edilir
        cv2.imwrite("ImageProcessingTrain/Maskless/"+str(i)+".jpg",flipped)



# veri seti eksik olan maskesiz insanların resimlerini 435'e tamamlamadık.
for i in range(241,436):

    #var olan resimden rastgele index seçilir.
    index = random.randint(0,len(imageNumberList)-1)

    #random sayı değeri
    print("random sayı değeri: ",imageNumberList[index]) 

    temp = 'ImageProcessingTrain/Maskless/'+str(imageNumberList[index])+'.jpg';

    #resim okuma işlemi yapılır.
    syntheticImg = cv2.imread(temp)

    #resim satır ve sutun değerleri alınır
    rows, cols = syntheticImg.shape[:2]

    #Veri setinin yarısı ile döndürme diğer yarısı ile simetri işlemi yapılır
    if i <= 340:
        #döndürme işlemi
        angle = random.randint(-45,45)
        rotation_matrix = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        img_rotation = cv2.warpAffine(syntheticImg,rotation_matrix,(cols,rows))

        #resim ekranda gösterilir.
        #cv2.imshow("dondurme" + str(i),img_rotation)

        #resim klasöre yazdırılır.
        cv2.imwrite("ImageProcessingTrain/Maskless/"+str(i)+".jpg",img_rotation)
        
       
    else:
        #simetri işlemi
        flipped = cv2.flip(syntheticImg,1) #horizontal image

        #resim ekranda gösterilir.
        #cv2.imshow("simetri" + str(i),flipped)

        #resim klasöre yazdırılır.
        cv2.imwrite("ImageProcessingTrain/Maskless/"+str(i)+".jpg",flipped)

    

cv2.waitKey(0)
cv2.destroyAllWindows()
