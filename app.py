import cv2

MODEL_PATH = './resources/models/face_detection.xml'
INPUT_PATH = './resources/photos/people-wedding.jpeg'
face_cascade = None
faces = None
image = None


def setupModelAndInput(input, model):
    ''' Takes input image path and model data path and updates face_cascade and image variables. '''

    global face_cascade
    global image
    face_cascade = cv2.CascadeClassifier(model)
    image = cv2.imread(input)


def detectFaces(image):
    ''' Returns the detected faces. '''

    if face_cascade != None:
        return face_cascade.detectMultiScale(image, 1.1, 4)


def drawBoxAroundFaces(faces):
    ''' Draws box around each face found then saves the result. '''

    if faces != None:
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imwrite('result.png', image)
        print('result saved')
    else:
        print('input cannot be None')


setupModelAndInput(INPUT_PATH, MODEL_PATH)
faces = detectFaces(image)
drawBoxAroundFaces(faces)
