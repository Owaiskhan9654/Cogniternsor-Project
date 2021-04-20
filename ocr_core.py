try:
    from PIL import Image
    import numpy as np
    import cv2
    import imutils
except ImportError:
    import Image


import pytesseract as pt



def ocr_core(filename):
    path = "static\\upload_car\\"

    image = cv2.imread(path +filename)

    image = imutils.resize(image, width=500)
          
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    edged = cv2.Canny(gray, 170, 200)
  
    
    cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    
    image1 = image.copy()
    cv2.drawContours(image1, cnts, -1, (0, 255, 0), 3)
    

   
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    NumberPlateCnt = None  
    
    image2 = image.copy()
    cv2.drawContours(image2, cnts, -1, (0, 255, 0), 3)
    
    
    
    count = 0
    idx = 7
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4: 
            NumberPlateCnt = approx  

            
            x, y, w, h = cv2.boundingRect(c)  
            new_image = gray[y:y + h, x:x + w]  
            cv2.imwrite('Cropped Images Text/' + str(idx) + '.png', new_image) 
            idx += 1

            break



    cv2.drawContours(image, [NumberPlateCnt], -1, (0, 255, 0), 3)

    pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    text = pt.pytesseract.image_to_string(new_image, lang='eng')

    return text


