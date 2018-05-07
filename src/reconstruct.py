import numpy as np
import cv2
import math
import operator
import tensorflow as tf

def compute(file):
    charachters = []

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

    def resize(img, maxVal):
        height, width = img.shape
        scale = 1.0/ max(height, width)
        size = maxVal
        if max(height, width)/5 < maxVal:
            size = max(height, width)/5
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

    def findSub(img, label):
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
                resized_image = resize(crop_img, 10)
                # output = cv2.inRange(resized_image, 1, 255)
                output = resized_image
                if (x2 - x) > 50 :
                    crop_erosion = erosion2[y:y2, x:x2]
                    cv2.imshow("image" + str(label), imgcolor2)
                    findSub(crop_erosion, str(label) + "_" + str(i))
                else:
                    charachters.append({'x':x, 'y':y, 'h': (y2-y), 'w':(x2-x), 'img': output})

    for i in range(0, len(boundingRectsH)):
        y = boundingRectsY[i]
        y2 = y + boundingRectsH[i]
        x = boundingRectsX[i]
        x2 = x + boundingRectsW[i]
        if (y2-y) + (x2-x) > 120:
            crop_img = thresh[y:y2, x:x2]
            findSub(crop_img, i)
        elif (y2-y) + (x2-x) > 20:
            crop_img = erosion[y:y2, x:x2]
            resized_image = resize(crop_img, 10)
            # output = cv2.inRange(resized_image, 1, 255)
            output = resized_image
            charachters.append({'x':x, 'y':y, 'h': (y2-y), 'w':(x2-x), 'img': output})


    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for char in charachters:
        cv2.rectangle(img, (char['x'],char['y']), (char['x']+char['w'], char['y']+char['h']), (0,0,255), 2)
    cv2.imwrite('/home/max/src/tensorfun/ocr/raw_charsFound.jpg', img)

    return charachters

def sortChars (chars, line_height, line_offset, number_of_lines):
    lines = [None]*number_of_lines
    for i in range(0,number_of_lines):
        lines[i] = []
    for char in chars:
        line = math.floor((char['y']-line_offset)/line_height)
        lines[int(line)].append(char)
    for i in range(len(lines)):
        line = lines[i]
        lines[i] = sorted(line, key=operator.itemgetter('x'))
    return lines

charachters = compute('/home/max/src/tensorfun/ocr/coverletter.jpeg')
lines = sortChars(charachters, 2576/22, 280, 22)
for i in range(0, len(lines)):
    line = lines[i]
    for k in range(0, len(line)):
        cv2.imwrite('/media/pics/reconstruct/char_' + str(i) + "_" + str(k) + ".jpeg", line[k]['img'])

def getBytes(img):
    np_image_data = np.asarray(img)
    np_image_data=cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
    np_final = np.concatenate(np_image_data,axis=0)
    return np_final


# PREDICT

x = tf.placeholder("float", [None, 100])
y = tf.placeholder("float", [None, 26])  # None is for infinite
W = tf.Variable(tf.zeros([100, 26]))
b = tf.Variable(tf.zeros([26]))
# LOGISTIC REGRESSION MODEL
actv = tf.nn.softmax(tf.matmul(x, W) + b)
decoder = tf.argmax(actv, axis=1)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

sess = tf.Session()
saver.restore(sess, "/media/pics/model.ckpt")
for i in range(0, len(lines)):
    line = lines[i]
    lineDecode = ""
    for k in range(0, len(line)):
        if k > 0 and line[k]['x'] - line[k-1]['x'] - line[k-1]['w'] > 10:
            lineDecode += " "
        lineDecode += ''.join(map(unichr, [sess.run(decoder, {x: [getBytes(lines[i][k]['img'])]})[0] + 65]))
    print lineDecode
