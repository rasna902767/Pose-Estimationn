import cv2
import time
import poseModule as pm

import os
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false"

cap = cv2.VideoCapture(0)
pTime = 0
detector = pm.poseDetector()

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = detector.findPose(frame, draw=True)
    lmList = detector.getPosition(frame, draw=True)

    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime != pTime else 0
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (70, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("image", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()