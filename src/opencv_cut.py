import numpy as np
import cv2
import math

def compute(file, letter):
    img = cv2.imread(file)
    # define range of blue color in HSV
    low = np.array([0,0,0])
    high_val = 150
    high = np.array([high_val, high_val, high_val])

    thresh = cv2.inRange(img, low, high)

    cv2.imwrite('/home/max/src/tensorfun/ocr/raw_thresh.jpg', thresh)

    kernel = np.ones((2,2),np.uint8)
    erosion = cv2.erode(thresh,kernel,iterations = 1)

    cv2.imwrite('/home/max/src/tensorfun/ocr/raw_erode.jpg', erosion)

    im2, contours, hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    imgcolor = cv2.cvtColor(erosion,cv2.COLOR_GRAY2BGR)

    cv2.drawContours(imgcolor, contours, -1, (0,255,0), 1)

    boundingRectsX = []
    boundingRectsY = []
    boundingRectsW = []
    boundingRectsH = []

    def aabb (x1, y1, w1, h1, x2, y2, w2, h2):
        return (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and h1 + y1 > y2)

    def merge (x1, y1, w1, h1, x2, y2, w2, h2):
        xf = min(x1, x2)
        yf = min(y1, y2)
        xabsf = max(x1 + w1, x2 + w2)
        yabsf = max(y1 + h1, y2 + h2)
        wf = xabsf - xf
        hf = yabsf - yf
        return xf, yf, wf, hf

    def addBounding(bRx, bRy, bRw, bRh, x, y, w, h):
        for i in range(0, len(bRx)):
            if aabb (bRx[i], bRy[i], bRw[i], bRh[i], x, y, w, h):
                xn, yn, wn, hn = merge (bRx[i], bRy[i], bRw[i], bRh[i], x, y, w, h)
                if wn + hn < 80 :
                    bRx[i] = xn
                    bRy[i] = yn
                    bRw[i] = wn
                    bRh[i] = hn
                    return
        bRx.append(x)
        bRy.append(y)
        bRw.append(w)
        bRh.append(h)


    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if w + h > 20:
            cv2.rectangle(imgcolor, (x,y), (x+w,y+h), (255,0,0), 2)
            addBounding(boundingRectsX, boundingRectsY, boundingRectsW, boundingRectsH, x, y, w, h)

    for i in range(0, len(boundingRectsH)):
        x = boundingRectsX[i]
        y = boundingRectsY[i]
        w = boundingRectsW[i]
        h = boundingRectsH[i]
        if h + w > 40:
            cv2.rectangle(imgcolor, (x,y), (x+w,y+h), (0,0,255), 1)

    cv2.imwrite('/home/max/src/tensorfun/ocr/raw_contours.jpg', imgcolor)

    # def getLetter(loc):
    #     if(loc < b1):
    #         return ''.join(map(unichr, [letterStart]))
    #     elif(loc < b2):
    #         return ''.join(map(unichr, [letterStart+32]))
    #     elif(loc < b3):
    #         return ''.join(map(unichr, [letterStart+1]))
    #     elif(loc < b4):
    #         return ''.join(map(unichr, [letterStart+33]))
    #     elif(loc < b5):
    #         return ''.join(map(unichr, [letterStart+2]))
    #     elif(loc < b6):
    #         return ''.join(map(unichr, [letterStart+34]))
    #     elif(loc < b7):
    #         return ''.join(map(unichr, [letterStart+3]))
    #     else:
    #         return ''.join(map(unichr, [letterStart+35]))

    def resize(img, maxVal):
        height, width = img.shape
        scale = 1.0/ max(height, width)
        size = maxVal
        if max(height, width)/1.2 < maxVal:
            size = max(height, width)/1.2
        newSize = (int(size*width*scale), int(size*height*scale))
        if newSize[0] < 1:
            newSize = (1, newSize[1])
        if newSize[1] < 1:
            newSize = (newSize[0], 1)
        resized_image = cv2.resize(img, newSize)
        empty = np.zeros((maxVal, maxVal), np.uint8)
        # combine = np.copyto(empty, resized_image, casting='unsafe', where=True)
        empty[tuple([slice(0, n) for n in resized_image.shape])] = resized_image
        # empty[:len(resized_image)] = resized_image
        combine = empty
        return combine


    def findSub(img, label, oy):
        kernel2 = np.ones((6,2),np.uint8)
        erosion2 = cv2.erode(img,kernel2,iterations = 1)
        im3, contours2, hierarchy2 = cv2.findContours(erosion2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        imgcolor2 = cv2.cvtColor(erosion2,cv2.COLOR_GRAY2BGR)
        bRx = []
        bRy = []
        bRw = []
        bRh = []
        for contour in contours2:
            x,y,w,h = cv2.boundingRect(contour)
            if w + h > 10:
                cv2.rectangle(imgcolor2, (x,y), (x+w,y+h), (255,0,0), 2)
                addBounding(bRx, bRy, bRw, bRh, x, y, w, h)
        for i in range(0, len(bRh)):
            x = bRx[i]
            y = bRy[i]
            w = bRw[i]
            h = bRh[i]
            if h + w > 40:
                cv2.rectangle(imgcolor2, (x,y), (x+w,y+h), (0,0,255), 1)
        for i in range(0, len(bRh)):
            y = bRy[i]
            y2 = y + bRh[i]
            x = bRx[i]
            x2 = x + bRw[i]
            # if (y2-y) + (x2-x) > 120:
            #     crop_img = img[y:y2, x:x2]
            #     findSub(crop_img, i)
            if (y2-y) + (x2-x) > 40:
                crop_img = img[y:y2, x:x2]
                resized_image = resize(crop_img, 28)
                # output = cv2.inRange(resized_image, 1, 255)
                output = resized_image
                if (x2 - x) > 50 :
                    crop_erosion = erosion2[y:y2, x:x2]
                    cv2.imshow("image" + str(label), imgcolor2)
                    findSub(crop_erosion, str(label) + "_" + str(i), oy)
                else:
                    cv2.imwrite('/media/pics/chars/' + letter + str(label) + "_" + str(i) + ".jpg", output)

    for i in range(0, len(boundingRectsH)):
        y = boundingRectsY[i]
        y2 = y + boundingRectsH[i]
        x = boundingRectsX[i]
        x2 = x + boundingRectsW[i]
        if (y2-y) + (x2-x) > 120:
            crop_img = thresh[y:y2, x:x2]
            findSub(crop_img, i, y)
        elif (y2-y) + (x2-x) > 20:
            crop_img = erosion[y:y2, x:x2]
            resized_image = resize(crop_img, 28)
            # output = cv2.inRange(resized_image, 1, 255)
            output = resized_image
            cv2.imwrite('/media/pics/chars/' + letter + str(i) + ".jpg", output)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

for z in range(0, 26):
    compute('/home/max/src/tensorfun/ocr/training_data/page_' + '{0:0>4}'.format(z) + ".jpg", ''.join(map(unichr, [65+z])))
    print "Done with " + ''.join(map(unichr, [65+z]))

# compute('/home/max/src/tensorfun/ocr/1.jpeg', 65,    558, 834, 1106, 1376, 1734, 2000, 2344)
# compute('/home/max/src/tensorfun/ocr/2.jpeg', 65+4*1,558, 834, 1106, 1376, 1665, 1935, 2233)
# compute('/home/max/src/tensorfun/ocr/3.jpeg', 65+4*2,558, 834, 1106, 1376, 1665, 1935, 2233)
# compute('/home/max/src/tensorfun/ocr/4.jpeg', 65+4*3,558, 834, 1106, 1376, 1665, 1935, 2233)
# compute('/home/max/src/tensorfun/ocr/5.jpeg', 65+4*4,558, 834, 1106, 1376, 1665, 1935, 2233)
# compute('/home/max/src/tensorfun/ocr/6.jpeg', 65+4*5,558, 834, 1106, 1376, 1665, 1935, 2233)
# compute('/home/max/src/tensorfun/ocr/7.jpeg', 65+4*6,558, 834, 1106, 1376, 1665, 1935, 2233)
