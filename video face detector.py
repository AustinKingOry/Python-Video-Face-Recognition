'''import cv2
from random import randrange
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#img=cv2.imread('image.jpg')
webcam = cv2.VideoCapture('justin.mp4')
while True:
    successful_frame_read, frame = webcam.read()
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    for(x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w, y+h),(randrange(128),randrange(128),randrange(128)),10)
    cv2.imshow('Austin face detector',frame)
    key = cv2.waitkey(1)
    #quit when Q or q is pressed
    if key==81 or key==113:
        break
webcam.release()
cv2.destroyAllWindows()'''
import cv2
from random import randrange
def draw_found_faces(detected, image, color: tuple):
    for (x, y, width, height) in detected:
        cv2.rectangle(
            image,
            (x, y),
            (x + width, y + height),
            color,
            thickness=2
        )

# Capturing the Video Stream
video_capture = cv2.VideoCapture('tiktok.mp4')

# Creating the cascade objects
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

while True:
    # Get individual frame
    _, frame = video_capture.read()
    # Covert the frame to grayscale
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
	# Detect all the faces in that frame
    detected_faces = face_cascade.detectMultiScale(image=grayscale_image, scaleFactor=1.3, minNeighbors=4)
    detected_eyes = eye_cascade.detectMultiScale(image=grayscale_image, scaleFactor=1.3, minNeighbors=4)
    draw_found_faces(detected_faces, frame, (randrange(128),randrange(128),randrange(128)))
    draw_found_faces(detected_eyes, frame, (randrange(128),randrange(128),randrange(128)))

    # Display the updated frame as a video stream
    cv2.imshow('Webcam Face Detection', frame)

    # Press the ESC key to exit the loop
    # 27 is the code for the ESC key
    if cv2.waitKey(1) == 113 or cv2.waitKey(1)==81:
        break

# Releasing the webcam resource
video_capture.release()

# Destroy the window that was showing the video stream
cv2.destroyAllWindows()
print('code completed')