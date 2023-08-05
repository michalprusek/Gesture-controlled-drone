import cv2 as cv
import sys
import mediapipe as mp
import time
import numpy as np
import PomocneFunkce as Pf


np.set_printoptions(threshold=sys.maxsize)


class FaceDetector:

    def __init__(self):
        self.mp_facedetector = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils

        self.cap = cv.VideoCapture(0)
        self.descriptors_num = 20
        self.kNN = 5
        self.probThreshold = 7

        self.descriptors2, self.gestures2 = Pf.zpracovat2("deskriptory2.npy", self.descriptors_num, novy=False)

        #Five,Four,Three,Two,One,Circle,Rocknroll,Thumb
        self.gesto = "Thumb"
        self.Gcount = len(self.descriptors2)
        self.saveArray = []
        self.pocetNafocenychGest = 0
        self.pocetSpravnych = 0
        self.pocetCelkovych = 0

    def run(self):
        with self.mp_facedetector.FaceDetection(min_detection_confidence=0.8) as face_detection:
            while self.cap.isOpened():
                success, img = self.cap.read()

                start = time.time()

                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

                results = face_detection.process(img)

                img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

                if results.detections:
                    bbox = results.detections[0].location_data.relative_bounding_box
                    Reye_keypoint = results.detections[0].location_data.relative_keypoints[0]
                    Leye_keypoint = results.detections[0].location_data.relative_keypoints[1]
                    mouth_keypoint = results.detections[0].location_data.relative_keypoints[3]

                    h, w, c = img.shape

                    boundingBox = [int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)]
                    mouth = [int(mouth_keypoint.x * w), int(mouth_keypoint.y * h)]
                    Reye = [int(Reye_keypoint.x * w - boundingBox[0] - 30),
                            int(Reye_keypoint.y * h - boundingBox[1] - 10)]
                    Leye = [int(Leye_keypoint.x * w - boundingBox[0] - 30),
                            int(Leye_keypoint.y * h - boundingBox[1] - 10)]

                    Xup, Yup, Xdown, Ydown = Pf.drawDetections(img, boundingBox, hand_window_size_mult=1.8)

                    face_cropped = img[np.abs(boundingBox[1] + 10):np.abs(mouth[1]) - 50,
                                   np.abs(boundingBox[0] + 30):np.abs(boundingBox[0] + boundingBox[2] - 30)]

                    hand_cropped = img[np.abs(Yup + 5):np.abs(Ydown - 5), np.abs(Xup + 5):np.abs(Xdown - 5)]

                    if hand_cropped.any():

                        hand_cropped_ycrcb = cv.cvtColor(hand_cropped, cv.COLOR_BGR2YCR_CB)

                        face_cropped_blur = cv.GaussianBlur(face_cropped, (5, 5), 0)

                        face_cropped_gray = cv.cvtColor(face_cropped_blur, cv.COLOR_BGR2GRAY)

                        face_cropped_otsu = Pf.otsu(face_cropped_gray)
                        face_cropped_ROIotsu = Pf.ROIotsu(face_cropped_gray, vertical_blocks=5, horizontal_blocks=4)

                        face_composed = np.clip(face_cropped_otsu + face_cropped_ROIotsu, 0, 1)

                        face_composed_erosion_eyes = Pf.Erosion(face_composed, 2)

                        radius = int(((np.shape(face_cropped)[0] + np.shape(face_cropped)[1]) / 2) / 5)
                        face_composed_erosion_eyes = cv.circle(face_composed_erosion_eyes, Reye, radius, 0, cv.FILLED, 8, 0)
                        face_composed_erosion_eyes = cv.circle(face_composed_erosion_eyes, Leye, radius, 0, cv.FILLED, 8, 0)

                        face_cropped_otsu_color = face_cropped.copy()
                        face_cropped_otsu_color[face_composed_erosion_eyes < 1] = [0, 0, 0]

                        face_cropped_otsu_ycrcb = cv.cvtColor(face_cropped_otsu_color, cv.COLOR_BGR2YCR_CB)


                        img_backprojected, prob = Pf.my_improved_backprojectionV3(face_cropped_otsu_ycrcb,
                                                                                  hand_cropped_ycrcb, threshold=self.probThreshold)

                        #cv.imshow("hand",img_backprojected)
                        drawing = np.zeros(hand_cropped.shape, np.uint8)

                        contours, hierarchy = cv.findContours(img_backprojected, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

                        if len(contours) > 0:
                            contour = max(contours, key=lambda x: cv.contourArea(x), default=0)
                            cv.drawContours(drawing, [contour], -1, (255, 255, 255), 0)
                            if len(contour) > self.descriptors_num:

                                FD = Pf.fourier_descriptors(contour, self.descriptors_num)

                                vyherniGesto = Pf.zkouska(contour, self.descriptors_num, self.descriptors2, self.gestures2, pocetSousedu=self.kNN)
                                vysledneGesto = str(vyherniGesto[0][0][:vyherniGesto[0][0].find("_")])
                                cv.putText(img, vysledneGesto, (Xup+10,Yup-15), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)


                cv.imshow("Image", img)
                end = time.time()
                totalTime = end - start
                # print(f"Total time: {totalTime}")


                fps = 1 / totalTime
                #print(f"FPS: {int(fps)}")
                cv.waitKey(1)
