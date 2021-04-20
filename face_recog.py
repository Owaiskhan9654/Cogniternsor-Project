import cv2
import face_recognition
def face_recog(filename,a):
    path="static\\upload_face\\"
    img_orignal = face_recognition.load_image_file(path+filename)
    img_orignal = cv2.cvtColor(img_orignal,cv2.COLOR_BGR2RGB)
    imgTest = face_recognition.load_image_file(a)
    imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
    faceLoc = face_recognition.face_locations(img_orignal)[0]
    encodesharukh = face_recognition.face_encodings(img_orignal)[0]
    cv2.rectangle(img_orignal,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
 
    faceLocTest = face_recognition.face_locations(imgTest)[0]
    encodeTest = face_recognition.face_encodings(imgTest)[0]
    cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
 
    results = face_recognition.compare_faces([encodesharukh],encodeTest)
    faceDis = face_recognition.face_distance([encodesharukh],encodeTest)

    cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    return (results, faceDis)
