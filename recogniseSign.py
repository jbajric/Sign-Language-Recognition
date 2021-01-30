import cv2
import numpy as np
from PIL import Image
import tensorflow as tf


def nothing(x):
    pass

image_x, image_y = 64, 64
classifier = tf.keras.models.load_model('model.h5')

def predictor():
    from keras.preprocessing import image
    test_image = image.load_img('image.png', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)

    if result[0][0] == 1:
        return 'A'
    elif result[0][1] == 1:
        return 'B'
    elif result[0][2] == 1:
        return 'C'
    elif result[0][3] == 1:
        return 'D'
    elif result[0][4] == 1:
        return 'E'
    elif result[0][5] == 1:
        return 'F'
    elif result[0][6] == 1:
        return 'G'
    elif result[0][7] == 1:
        return 'H'
    elif result[0][8] == 1:
        return 'I'
    elif result[0][9] == 1:
        return 'J'
    elif result[0][10] == 1:
        return 'K'
    elif result[0][11] == 1:
        return 'L'
    elif result[0][12] == 1:
        return 'M'
    elif result[0][13] == 1:
        return 'N'
    elif result[0][14] == 1:
        return 'O'
    elif result[0][15] == 1:
        return 'P'
    elif result[0][16] == 1:
        return 'Q'
    elif result[0][17] == 1:
        return 'R'
    elif result[0][18] == 1:
        return 'S'
    elif result[0][19] == 1:
        return 'T'
    elif result[0][20] == 1:
        return 'U'
    elif result[0][21] == 1:
        return 'V'
    elif result[0][22] == 1:
        return 'W'
    elif result[0][23] == 1:
        return 'X'
    elif result[0][24] == 1:
        return 'Y'
    elif result[0][25] == 1:
        return 'Z'


cam = cv2.VideoCapture(0)

#Define trackbars for manipulating mask
cv2.namedWindow("Sliders for HSV on mask")
cv2.resizeWindow("Sliders for HSV on mask", 400, 350)
cv2.createTrackbar("Low - H", "Sliders for HSV on mask", 0, 179, nothing)
cv2.createTrackbar("Low - S", "Sliders for HSV on mask", 0, 255, nothing)
cv2.createTrackbar("Low - V", "Sliders for HSV on mask", 0, 255, nothing)
cv2.createTrackbar("High - H", "Sliders for HSV on mask", 0, 179, nothing)
cv2.createTrackbar("High - S", "Sliders for HSV on mask", 0, 255, nothing)
cv2.createTrackbar("High - V", "Sliders for HSV on mask", 0, 255, nothing)

img_counter = 0
img_text = ''

imagePresent = False

while True:
    # Show picture of the sign language alphabet
    if imagePresent == False:
        imagePresent = True
        imageSLA = Image.open('SL_alphabet.png')
        imageSLA.show()

    # Create main window for recognition
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    low_h = cv2.getTrackbarPos("Low - H", "Sliders for HSV on mask")
    low_s = cv2.getTrackbarPos("Low - S", "Sliders for HSV on mask")
    low_v = cv2.getTrackbarPos("Low - V", "Sliders for HSV on mask")
    high_h = cv2.getTrackbarPos("High - H", "Sliders for HSV on mask")
    high_s = cv2.getTrackbarPos("High - S", "Sliders for HSV on mask")
    high_v = cv2.getTrackbarPos("High - V", "Sliders for HSV on mask")

    img = cv2.rectangle(frame, (425, 100), (625, 300), (255, 0, 0), thickness=2, lineType=8, shift=0)

    low_blue = np.array([low_h, low_s, low_v])
    high_blue = np.array([high_h, high_s, high_v])
    imcrop = img[102:298, 427:623]
    hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low_blue, high_blue)

    # Display recognized sign as a character
    cv2.putText(frame, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0, 0))
    cv2.imshow("Main window for sign recognition", frame)
    cv2.imshow("Mask", mask)

    img_name = "image.png"
    save_img = cv2.resize(mask, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    img_text = predictor()

    if cv2.waitKey(1) == 27:
        break


cam.release()
cv2.destroyAllWindows()