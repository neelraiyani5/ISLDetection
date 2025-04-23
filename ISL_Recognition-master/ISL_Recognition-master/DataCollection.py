import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
detector = HandDetector(maxHands=2)  # Changed to maxHands=2 to detect up to two handsS
offset = 20
imgSize = 300
folder = "dataset/R"  # Updated folder name for Indian Sign Language
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        # Create a blank white image for the output
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        if len(hands) == 1:  # Single hand case
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Add padding with offset
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

            # Add padding with offset
            imgCrop = img[max(0, minY - offset):min(img.shape[0], maxY + offset),
                      max(0, minX - offset):min(img.shape[1], maxX + offset)]

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

                cv2.imshow("ImageCrop", imgCrop)

        cv2.imshow("ImageWhite", imgWhite)

        # Display the number of hands detected
        hand_text = f"Hands: {len(hands)}"
        cv2.putText(img, hand_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord("z"):
        counter += 1
        # Save with indication of how many hands were used
        hands_count = len(hands) if hands else 0
        cv2.imwrite(f'{folder}/Image_{hands_count}hand_{time.time()}.jpg', imgWhite)
        print(f"Saved image {counter} with {hands_count} hand(s)")
    elif key == ord("q"):  # Added quit option
        break

cap.release()
cv2.destroyAllWindows()