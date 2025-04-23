import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # Changed to maxHands=2 for ISL
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]  

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        if len(hands) == 1:  # Single hand case
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Add padding with offset and ensure within image boundaries
            imgCrop = img[max(0, y - offset):min(img.shape[0], y + h + offset),
                      max(0, x - offset):min(img.shape[1], x + w + offset)]

            if imgCrop.size != 0:  # Check if cropped image is not empty
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                # Draw rectangle and label for single hand
                cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                              (x - offset + 90, y - offset), (255, 0, 255), cv2.FILLED)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (255, 0, 255), 4)

                cv2.imshow("ImageCrop", imgCrop)

        else:  # Two hands case
            # Find bounding box that encompasses both hands
            minX = min(hands[0]['bbox'][0], hands[1]['bbox'][0])
            minY = min(hands[0]['bbox'][1], hands[1]['bbox'][1])

            # Calculate width and height to include both hands
            maxX = max(hands[0]['bbox'][0] + hands[0]['bbox'][2],
                       hands[1]['bbox'][0] + hands[1]['bbox'][2])
            maxY = max(hands[0]['bbox'][1] + hands[0]['bbox'][3],
                       hands[1]['bbox'][1] + hands[1]['bbox'][3])

            w = maxX - minX
            h = maxY - minY
            x, y = minX, minY

            # Add padding with offset and ensure within image boundaries
            imgCrop = img[max(0, y - offset):min(img.shape[0], y + h + offset),
                      max(0, x - offset):min(img.shape[1], x + w + offset)]

            if imgCrop.size != 0:  # Check if cropped image is not empty
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                # Draw rectangle around both hands
                cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                              (x - offset + 90, y - offset), (255, 0, 255), cv2.FILLED)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (255, 0, 255), 4)

                cv2.imshow("ImageCrop", imgCrop)

        # Get prediction from the model
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        print(prediction, index)

        # Display the prediction
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

        # Display number of hands detected
        hand_text = f"Hands: {len(hands)}"
        cv2.putText(imgOutput, hand_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)

    if key == ord("q"):  # Added quit option
        break

cap.release()
cv2.destroyAllWindows()