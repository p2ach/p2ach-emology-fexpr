import mediapipe as mp
import cv2
import numpy as np


class FaceRun():
    def  __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.offsets=(0, 0)

    def get_detection(self,img,frame_width,frame_height):
        with self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5) as face_detection:
            try:
                image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                return None
            else:
                results = face_detection.process(image)
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.detections:
                    for detection in results.detections:
                        x1, x2, y1, y2 = self.apply_offsets(detection.location_data, frame_height, frame_width, self.offsets)
                        face_img = image[y1:y2, x1:x2]
                        return face_img
                else:
                    return None

    def get_landmark(self,face_img):
        with self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            results = face_mesh.process(face_img)
            # print("results",results)
            landmark = []
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for lnd in face_landmarks.landmark:
                        tmp_lnd=[float(lnd.x)]
                        tmp_lnd.append(float(lnd.y))
                        tmp_lnd.append(float(lnd.z))
                        landmark.append(np.array(tmp_lnd).astype(np.float32))
                        # print("lnd",lnd)
                return np.array(landmark).astype(np.float32)
            else:
                return None

                    #
                    # print(a,a)




    def apply_offsets(self,face_coordinates, frame_height, frame_width, offsets):
        x = max(int(face_coordinates.relative_bounding_box.xmin * frame_width), 0)
        y = max(int(face_coordinates.relative_bounding_box.ymin * frame_height), 0)
        width = min(int(face_coordinates.relative_bounding_box.width * frame_width), frame_width)
        height = min(int(face_coordinates.relative_bounding_box.height * frame_height), frame_height)
        x_off, y_off = offsets
        return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)











