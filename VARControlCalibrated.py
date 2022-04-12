import math
import cv2
import numpy as np
import time
import pyautogui as pyautogui
import cv2
import mediapipe as mp
import time
import math

from VideoCaptureModule import VCModule
class faceDetector():
    def __init__(self):
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(0.75)
    def findFace(self,img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        if self.results.detections:
            for detection in self.results.detections:
                self.mp_drawing.draw_detection(img, detection)
        return img
    def detectFace(self,img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        if self.results.detections:
            return True
        else:
            return False
class handDetector():
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.static_image_mode, self.max_num_hands,
        self.min_detection_confidence, self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True, flipType=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
        if draw:
            return allHands, img
        else:
            return allHands

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = [0,0,0,0]
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
        else:
            return self.lmList, bbox
        for id, lm in enumerate(myHand.landmark):
            # print(id, lm)
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            xList.append(cx)
            yList.append(cy)
            # print(id, cx, cy)
            self.lmList.append([id, cx, cy])
            if draw:
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        xmin, xmax = min(xList), max(xList)
        ymin, ymax = min(yList), max(yList)
        bbox = xmin, ymin, xmax, ymax

        if draw:
            cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
        (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)
        
        return self.lmList, bbox

    def fingersUp(self, myHand):
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:
            fingers = []
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True):

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length, info, img
        else:
            return length, info


##########################
# wCam, hCam = 1280, 720
wCam, hCam = 640, 480
frameR = 100 # Frame Reduction
smoothening = 7
isCommandOn = True
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
#########################

 
cap = VCModule().getVideoCapture()
cap.set(3, wCam)
cap.set(4, hCam)
detector = handDetector(max_num_hands=2)
fdetector = faceDetector()
wScr, hScr = pyautogui.size()
leftMouseDown = False
rightMouseDown = False
x1, y1, x2, y2 = 0, 0, 0, 0
isFace = False

x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2) 
currentDistance = 0 
showHand = False
while True:
    success, img = cap.read()
    isFace = fdetector.detectFace(img)
    allHands, img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    if len(allHands) != 0:
        showHand = True
    if len(allHands) == 2:
        if detector.fingersUp(allHands[0]) == [1, 1, 0, 0, 0] and \
                detector.fingersUp(allHands[1]) == [1, 1, 0, 0, 0]:
            lmList1 = allHands[0]["lmList"]
            lmList2 = allHands[1]["lmList"]
            if startDist is None:
                length, info, img = detector.findDistance(allHands[0]["center"], allHands[1]["center"], img)
                startDist = length

            length, info, img = detector.findDistance(allHands[0]["center"], allHands[1]["center"], img)
            scale = int((length - startDist) // 2)
            cx, cy = info[4:]
            if length < startDist and length < startDist*3//4:
                if isCommandOn:
                    pyautogui.hotkey('ctrl', '-')
                startDist = startDist*3//4
            if length > startDist and length > startDist*5//4:
                if isCommandOn:
                    pyautogui.hotkey('ctrl', '+')
                startDist = startDist*5//4
        else:
            startDist = None

    if len(allHands) == 1:
        if len(lmList) != 0:
            x1, y1 = lmList[5][1:]
            x2, y2 = lmList[4][1:]
            x4, y4, z4 = lmList[5]
            x5, y5, y5 = lmList[17]
        x, y, w, h = bbox
        distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
        A, B, C = coff
        distanceCM = A * distance ** 2 + B * distance + C
        if currentDistance > 100:
            currentDistance = distanceCM
        fingers = detector.fingersUp(allHands[0])
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
        (255, 0, 255), 2)
        if isFace == False and fingers[0] == 0 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            quit()
        if showHand == True and fingers[0] == 0 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            quit()
        if fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
            rightLength, rightLineInfo, rightImg = detector.findDistance(lmList[8][1:], lmList[12][1:], img)
            leftLength, LeftLineInfo, leftImg = detector.findDistance(lmList[8][1:], lmList[4][1:], img)
            if leftLength < 50:
                cv2.circle(leftImg, (LeftLineInfo[4], LeftLineInfo[5]),
                15, (0, 255, 0), cv2.FILLED)
                leftMouseDown = True
            elif leftLength >= 50:  
                if leftMouseDown == True:
                    if isCommandOn:
                        pyautogui.click(button="primary")
                leftMouseDown = False
                currentDistance = distanceCM

            if rightLength < 30:
                cv2.circle(leftImg, (rightLineInfo[4], rightLineInfo[5]),
                15, (0, 255, 0), cv2.FILLED)
                if isCommandOn:
                    pyautogui.click(button="secondary")

        x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
        y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening
        
        pyautogui.moveTo(wScr - clocX, clocY)

    cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
    plocX, plocY = clocX, clocY

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
    (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)