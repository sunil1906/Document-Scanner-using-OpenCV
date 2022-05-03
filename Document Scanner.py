import cv2
import numpy as np

imageCnt = 0
heightImg = 640
widthImg  = 480
total_images = 7
flag = 1

## TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray,scale,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver


def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area

def drawRectangle(img,biggest,thickness):
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)

    return img

def reorder(myPoints):

    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew

def getPerspectiveTransform(sourcePoints, destinationPoints):
    """
    Calculates the 3x3 matrix to transform the four source points to the four destination points
    /* Calculates coefficients of perspective transformation
    * which maps soruce (xi,yi) to destination (ui,vi), (i=1,2,3,4):
    *
    *      c00*xi + c01*yi + c02
    * ui = ---------------------
    *      c20*xi + c21*yi + c22
    *
    *      c10*xi + c11*yi + c12
    * vi = ---------------------
    *      c20*xi + c21*yi + c22
    *
    * Coefficients are calculated by solving linear system:
    *             a                         x    b
    * / x0 y0  1  0  0  0 -x0*u0 -y0*u0 \ /c00\ /u0\
    * | x1 y1  1  0  0  0 -x1*u1 -y1*u1 | |c01| |u1|
    * | x2 y2  1  0  0  0 -x2*u2 -y2*u2 | |c02| |u2|
    * | x3 y3  1  0  0  0 -x3*u3 -y3*u3 |.|c10|=|u3|,
    * |  0  0  0 x0 y0  1 -x0*v0 -y0*v0 | |c11| |v0|
    * |  0  0  0 x1 y1  1 -x1*v1 -y1*v1 | |c12| |v1|
    * |  0  0  0 x2 y2  1 -x2*v2 -y2*v2 | |c20| |v2|
    * \  0  0  0 x3 y3  1 -x3*v3 -y3*v3 / \c21/ \v3/
    *
    * where:
    *   cij - matrix coefficients, c22 = 1
    */

    """
    
    if sourcePoints.shape != (4,2) or destinationPoints.shape != (4,2):
        raise ValueError("There must be four source points and four destination points")
    
    
    a = np.zeros((8, 8))
    b = np.zeros((8))
    for i in range(4):
        a[i][0] = a[i+4][3] = sourcePoints[i][0]
        a[i][1] = a[i+4][4] = sourcePoints[i][1]
        a[i][2] = a[i+4][5] = 1
        a[i][3] = a[i][4] = a[i][5] = 0
        a[i+4][0] = a[i+4][1] = a[i+4][2] = 0
        a[i][6] = -sourcePoints[i][0]*destinationPoints[i][0]
        a[i][7] = -sourcePoints[i][1]*destinationPoints[i][0]
        a[i+4][6] = -sourcePoints[i][0]*destinationPoints[i][1]
        a[i+4][7] = -sourcePoints[i][1]*destinationPoints[i][1]
        b[i] = destinationPoints[i][0]
        b[i+4] = destinationPoints[i][1]

    x = np.linalg.solve(a, b)
    x.resize((9,), refcheck=False)
    x[8] = 1 # Set c22 to 1 as indicated in comment above
    return x.reshape((3,3))

print('Press "n" for Next Image and press "q" to quit')
while True:
    imagePath = 'image ' + str(imageCnt) + '.jpeg'
    img = cv2.imread(imagePath)

    # Resizing image
    img = cv2.resize(img, (widthImg, heightImg))
    # Color to Gray
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Gaussian Blur
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    # Canny Edge detection
    thres = [50, 200]
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])   
    
    # Finding all contours
    imgContours = img.copy()
    imgBigContour = img.copy()
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)
    
    # Finding Biggest Contour
    biggest, maxArea = biggestContour(contours)

    if biggest.size != 0:
        biggest = reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)
        imgBigContourWithBorder = imgBigContour.copy()
        imgBigContourWithBorder = drawRectangle(imgBigContourWithBorder, biggest, 2)
        temp = [i[0] for i in biggest]
        biggest = temp
        src = np.float32(biggest)
        dest = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = getPerspectiveTransform(src, dest)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    if flag == 1:
        flag = 0
        # Uncomment to see tranformation matrix
        # print("Matrix: \n", matrix)

    # Displaying the Intermediate outputs and Final Output images
    imageArray = ([img, imgGray, imgBlur, imgThreshold], [imgContours, imgBigContour, imgBigContourWithBorder, imgWarpColored])
    labels = [["Original", "Grayscale", "Blurred", "Threshold"], ["All Contours", "Biggest Contour", "ROI", "Warped Image"]]
    stackedImage = stackImages(imageArray, 0.55, labels)
    cv2.imshow("Result", stackedImage) 

    # Check for exit
    keyPressed = cv2.waitKey(1) & 0xFF
    if keyPressed == ord('q'):
        cv2.destroyAllWindows()
        break
    if keyPressed == ord('n'):
        imageCnt = (imageCnt + 1) % total_images
        flag = 1
       
