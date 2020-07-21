import cv2
import re
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from imutils import contours
from datetime import datetime
from data_access_layer import retrieve_data_from_firebase,save_guestUser_to_firebase
from data_access_layer_cloud_firestore import setTrackedUsers
#from readTextApiVersion import visionApi

plate_obj = {} 

# Validate Number plate text
def validateNumberPlate(text):
      validator = re.compile('^[a-zA-Z]{2,3}\-\d{4}$')
      mo = validator.search(text)
      if not (mo is None):
        print(mo.string)
        return mo.string
      else:
        #visionApi()
        print('****** Error in validating number plate read value ******')
        return text


def drawContoursEachCharater():
    import numpy as np
    import cv2
    plate = ''
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
        if cv2.contourArea(contours[i]) > 500:
            x,y,w,h = cv2.boundingRect(contours[i])
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.imwrite("contour.jpg", image)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    data = pytesseract.image_to_string(image, lang='eng', config='--psm 6')           
    cv2.imshow('sample',image)
    cv2.imshow('grayImage',grayImage)
    cv2.imshow('dst',dst)
    cv2.imwrite("grayImage.jpg", grayImage)
    print(plate,'scene',data)
    #visionApi()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test  number plate text using tesseract
def read_numberPlate():
  image = cv2.imread('testout.png')
  height, width, _ = image.shape
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (5,5), 0)
  thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
  cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  cnts, _ = contours.sort_contours(cnts, method="left-to-right")
  plate = ""
  for c in cnts:
    area = cv2.contourArea(c)
    x,y,w,h = cv2.boundingRect(c)
    center_y = y + h/2
    if area > 1000 and (w > h) and center_y > height/2:
      ROI = image[y:y+h, x:x+w]
      data = pytesseract.image_to_string(ROI, lang='eng', config='--psm 6')
      plate += data
      print(plate,"palets")
  plate = plate.split(" ")
  plate = plate[len(plate)-2] + plate[len(plate) - 1]
  setTrackedUsers(plate)
  validateNumberPlate(plate)
  print(plate,"palet",setTrackedUsers(plate))

# Read number plate text using tesseract
def read_numberPlate2(image):
  #drawContoursEachCharater()   
  height, width, _ = image.shape
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (5,5), 0)
  thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
  cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  cnts, _ = contours.sort_contours(cnts, method="left-to-right")
  plate = ""
  for c in cnts:
    area = cv2.contourArea(c)
    x,y,w,h = cv2.boundingRect(c)
    center_y = y + h/2
    if area > 1000 and (w > h):
      ROI = image[y:y+h, x:x+w]
      data = pytesseract.image_to_string(ROI, lang='eng', config='--psm 6')
      plate += data
  print(plate,"palet")
  plate = plate.split(" ")
  plate = plate[len(plate)-2] + "-"+ plate[len(plate) - 1]
  print(plate,"palet")
  plate = validateNumberPlate(plate)
  return (plate,setTrackedUsers(plate))


#drawContoursEachCharater()
#testing()
#validateNumberPlate('ABC-1234')
#read_numberPlate()