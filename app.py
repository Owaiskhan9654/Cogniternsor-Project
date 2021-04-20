import os

from flask import Flask, render_template, request

from ocr_core import ocr_core
from face_recog import face_recog
from Mask_Detection.MaskDetection import mask_detection
import base64
import numpy as np
import time
import cv2
import os , random

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/')
def base_page():
    logo ='/static/logo.svg'
    return render_template('base.html',logo=logo)

upload_face = '/static/upload_face/'
@app.route('/face_recog', methods=['GET', 'POST'])
def face_recogition():
    picFolder = os.path.join('static', 'upload_face')
    app.config['upload_face'] = picFolder
    img_src1 = os.path.join(app.config['upload_face'], 'Sharukh.jpg')
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('error.html', msg='No file selected')
        file = request.files['file']
        if file.filename == '':
            return render_template('error.html', msg='No file selected')

        if file and allowed_file(file.filename):
            file.save(os.path.join(os.getcwd() + upload_face, file.filename))

        a = picFolder + '\\Sharukh.jpg'

        condition, percentage = face_recog(file.filename, a)
        if condition==[True]:
            condition='Matching'
        elif condition==[False]:
            condition='Not Matching'
        return render_template('face_Detection.html', msg='Successfully processed',
                               condition=condition, percentage=percentage, img_src1=img_src1,
                               img_src2=upload_face + file.filename)
    elif request.method == 'GET':
        return render_template('face_Detection.html', img_src1=img_src1)

upload_car = '/static/upload_car/'


@app.route('/Car_plate', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('error.html', msg='No file selected')
        file = request.files['file']
        if file.filename == '':
            return render_template('error.html', msg='No file selected')

        if file and allowed_file(file.filename):
            file.save(os.path.join(os.getcwd() + upload_car, file.filename))
            extracted_text = ocr_core(file.filename)
            if extracted_text == '':
                return render_template('error.html')

            return render_template('Car_plate_Detection.html',
                                   msg='Successfully processed',
                                   extracted_text=extracted_text,
                                   img_src=upload_car + file.filename)
    elif request.method == 'GET':
        return render_template('Car_plate_Detection.html')


### Object detection module starts from here

route = os.path.dirname(os.path.abspath(__file__))
destination = os.path.join(route, "static/object_detectected_images", "test.jpeg")




@app.route('/object_detection')
def home():
    logo = '/static/logo_white.svg'
    return render_template('object_home.html',logo=logo)


@app.route('/camera')
def Camera():
    return render_template('object_cam.html')


@app.route('/objectsaveimage', methods=['POST'])
def saveImage():
    data_url = request.values['imageBase64']
    image_encoded = data_url.split(',')[1]
    body = base64.b64decode(image_encoded.encode('utf-8'))
    file = open(destination, "wb")
    file.write(body)
    return "ok"


@app.route('/object_process')
def process():
    return render_template('object_process.html')


@app.route('/objectshowimage')
def showImage():

    return render_template('object_output.html')


@app.route('/object_output')
def output():
    dam = 0
    # load the COCO class labels our YOLO model was trained on
    labelsPath = "object_modules/coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")
    # initialize a list of colors to represent each possible class label
    np.random.seed(11)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")
    # derive the paths to the YOLO weights and model configuration
    weightsPath = "object_modules/yolov3.weights"
    configPath = "object_modules/yolov3.cfg"
    # load our YOLO object detector trained on COCO dataset (80 classes)
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    # load our input image and grab its spatial dimensions
    # ************************************\
    image = cv2.imread("static/object_detectected_images/test.jpeg")
    # ************************************
    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    # show timing information on YOLO

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.5:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
                            0.3)
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

    a = random.randrange(1, 5000, 1)
    b = str(a) + ".jpeg"

    destination = os.path.join(route, "static/object_detectected_images", b)
    cv2.imwrite(destination, image)
    imgval = "../static/object_detectected_images/{}".format(b)

    return render_template('object_output.html', imgval=imgval)


upload_mask = '/static/upload_mask/'

# Face Detection module starts from Here
@app.route('/mask',methods=['GET', 'POST'])
def mask_detection1():
        if request.method == 'POST':
            if 'file' not in request.files:
                return render_template('error.html', msg='No file selected')
            file = request.files['file']
            if file.filename == '':
                return render_template('error.html', msg='No file selected')

            if file and allowed_file(file.filename):
                file.save(os.path.join(os.getcwd() + upload_mask, file.filename))

            label = mask_detection(file.filename)

            return render_template('Mask_Detection.html',msg='Successfully processed',
                                   label=label,img_src=upload_mask+file.filename)
        elif request.method == 'GET':
           return render_template('mask_Detection.html')


if __name__ == '__main__':
    app.run(debug=True)
