# import cv2
# import numpy as np
# import tesserocr as tr
# from PIL import Image


# image = cv2.imread("testout.png")

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', image)

# thresh = cv2.adaptiveThreshold(gray, 250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 1)
# cv2.imshow('thresh', thresh)

# kernel = np.ones((1, 1), np.uint8)
# img_dilation = cv2.dilate(thresh, kernel, iterations=1)

# ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

# clean_plate = 255 * np.ones_like(img_dilation)

# for i, ctr in enumerate(sorted_ctrs):
#     x, y, w, h = cv2.boundingRect(ctr)

#     roi = img_dilation[y:y + h, x:x + w]

#     # these are very specific values made for this image only - it's not a factotum code
#     if h > 10 and w > 10:
#         rect = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         clean_plate[y:y + h, x:x + w] = roi
#         cv2.imshow('ROI', rect)
#         cv2.imwrite('roi.png', roi)

# img = cv2.imread("roi.png")

# blur = cv2.medianBlur(img, 1)
# cv2.imshow('4 - blur', img) 
# cv2.waitKey(0)
# cv2.destroyAllWindows() 

# pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# api = tr.PyTessBaseAPI()

# try:
#     api.SetImage(pil_img)
#     boxes = api.GetComponentImages(tr.RIL.TEXTLINE, True)
#     text = api.GetUTF8Text()

# finally:
#     api.End()

# # clean the string a bit
# text = str(text).strip()

# print(text)
# cv2.waitKey(0)



def testing():
    import numpy as np
    import cv2

    image = cv2.imread("testout.png")
    image = cv2.resize(image, (400, 100), interpolation = cv2.INTER_AREA)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows,cols = grayImage.shape
    for i in range(rows):
        for j in range(cols):
            if grayImage[i,j] > 50:
                image[i,j] = 255
            else:
                image[i,j] = 0
    kernel = np.ones((5,5), np.uint8)
    kernel2 = np.ones((3,3), np.uint8)

    image = cv2.erode(image, kernel, iterations=1) 
    image = cv2.dilate(image, kernel2, iterations=1) 

    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = cv2.Canny(grayImage, 50, 100)
   
    #ret, bw_img = cv2.threshold(dst,127,255,cv2.THRESH_BINARY)

    #dst = cv2.morphologyEx(grayImage, cv2.MORPH_RECT, np.zeros((5,5), np.uint8),iterations=1)
    contours, heirarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    for i in range(0, len(contours)):
        print(cv2.contourArea(contours[i]))
        if cv2.contourArea(contours[i]) > 100:
            x,y,w,h = cv2.boundingRect(contours[i])
        # The w constrain to remove the vertical lines
            if w > 10:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.imwrite("contour.jpg", image)

    cv2.imshow('sample',image)
    cv2.imshow('grayImage',grayImage)
    cv2.imshow('dst',dst)
   

    cv2.waitKey(0)
    cv2.destroyAllWindows()

testing()