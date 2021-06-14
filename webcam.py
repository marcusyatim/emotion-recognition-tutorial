"""
This module is the main module in this package. It loads emotion recognition model from a file,
shows a webcam image, recognizes face and it's emotion and draw emotion on the image.
"""
import cv2
from face_detect import find_faces
from cv2 import WINDOW_NORMAL

def show_webcam_and_run(model, window_size=None, window_name='webcam', update_time=10):
    """
    Shows webcam image, detects faces and its emotions in real time
    :param model: Learnt emotion detection model.
    :param window_size: Size of webcam image window.
    :param window_name: Name of webcam image window.
    :param update_time: Image update time interval.
    """
    # CV parameters
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    font_scale = 1
    thickness = 8

    cv2.namedWindow(window_name, WINDOW_NORMAL)
    if window_size:
        width, height = window_size
        cv2.resizeWindow(window_name, width, height)

    vc = cv2.VideoCapture(0)
    if vc.isOpened():
        read_value, webcam_image = vc.read()
    else:
        print("webcam not found")
        return

    while read_value:
        for normalized_face, (x, y, w, h) in find_faces(webcam_image):
            # Do prediction
            prediction = model.predict(normalized_face)  
            prediction = prediction[0]
            emotions = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
            predicted_emotion = emotions[prediction]
            print (predicted_emotion)
            webcam_image = cv2.putText(webcam_image, str(predicted_emotion), (x,y-thickness//2), font, font_scale, (255, 255, 0), 1,lineType=cv2.LINE_AA)
        cv2.imshow(window_name, webcam_image)
        read_value, webcam_image = vc.read()
        key = cv2.waitKey(update_time)

        # Exit on ESC
        if key == 27:  
            break

    cv2.destroyWindow(window_name)


if __name__ == '__main__':
    # load model
    model = cv2.face.FisherFaceRecognizer_create()
    model.read("../Data/Models/emotion_detection_model_160.xml")

    # use learnt model
    window_name = 'WEBCAM (press ESC to exit)'
    show_webcam_and_run(model, window_size=(1600, 1200), window_name=window_name, update_time=8)