import cv2
import numpy as np
import pytesseract
import os

# C:\Program Files\Tesseract-OCR

per = 25

roi = [[(262, 194), (688, 218), 'text', 'kpb'],
       [(300, 228), (686, 242), 'text', 'kpd'],
       [(178, 290), (682, 304), 'text', 'nama'],
       [(254, 314), (412, 332), 'text', 'kp'],
       [(556, 314), (684, 332), 'text', 'warganegara'],
       [(378, 340), (686, 358), 'text', 'nopassport'],
       [(188, 368), (686, 386), 'text', 'alamat'],
       [(214, 394), (388, 410), 'text', 'notel'],
       [(468, 392), (620, 412), 'text', 'emel'],
       [(376, 420), (686, 438), 'text', 'jenisK'],
       [(234, 446), (294, 462), 'text', 'jumlahorg'],
       [(392, 472), (686, 488), 'text', 'tarikhMR'],
       [(408, 496), (686, 514), 'text', 'tarikhBR'],
       [(310, 524), (688, 538), 'text', 'alamatDes'],
       [(102, 574), (682, 594), 'text', 'sebabPP']]

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

imgQ = cv2.imread('Borang.png')
h, w, c = imgQ.shape
# imgQ = cv2.resize(imgQ, (w // 2, h // 2))

# create detector (ORB) - free to use
orb = cv2.ORB_create(1050)
kp1, des1 = orb.detectAndCompute(imgQ, None)
# imgKp1 = cv2.drawKeypoints(imgQ, kp1, None)

path = 'UserForms'
myPicList = os.listdir(path)
print(myPicList)
for j, y in enumerate(myPicList):
    img = cv2.imread(path + "/" + y)
    # img = cv2.resize(img, (w // 2, h // 2))
    # cv2.imshow(y, img)
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1)
    good = matches[:int(len(matches) * (per / 100))]
    imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:100], None, flags=2)

    # cv2.imshow(y, imgMatch)

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w, h))

    # cv2.imshow(y, imgScan)
    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    # myData = []

    print(f'################## Extracting Data from Form {j} ##################')

    for x, r in enumerate(roi):
        cv2.rectangle(imgMask, ((r[0][0]), r[0][1]), ((r[1][0]), r[1][1]), (0, 255, 0), cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)

        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        cv2.imshow(str(x), imgCrop)

imgShow = cv2.resize(imgShow, (w // 2, h // 2))
cv2.imshow(y + "2", imgShow)
cv2.waitKey(0)