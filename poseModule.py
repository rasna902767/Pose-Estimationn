import cv2
import mediapipe as mp
import time

class poseDetector():
    def __init__(self, mode=False, upBody=True,
                 smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.results = None

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon)

    def findPose(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks,
                                       self.mpPose.POSE_CONNECTIONS)
        return frame

    def getPosition(self, frame, draw=True):
        lmList = []
        HIGHLIGHT_IDS = {13, 14, 15, 16, 25, 27, 28, 26}  # elbows and wrists
        HIGHLIGHT_COLOR = (0, 0, 255)  # red (BGR)
        NORMAL_COLOR = (0, 255, 0)

        if self.results is not None and self.results.pose_landmarks:
            h, w, c = frame.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    color = HIGHLIGHT_COLOR if id in HIGHLIGHT_IDS else NORMAL_COLOR
                    cv2.circle(frame, (cx, cy), 5, color, cv2.FILLED)

        return lmList


def main():
    cap = cv2.VideoCapture('videos/6.mp4')
    if not cap.isOpened():
        print("Could not open video. Check the path.")
        return

    pTime = 0
    detector = poseDetector()

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = detector.findPose(frame, draw=True)
        lmList = detector.getPosition(frame, draw = True)
        if len(lmList) > 14:
            print(lmList[14])

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


if __name__ == "__main__":
    main()