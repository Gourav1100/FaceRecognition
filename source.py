# import modules
from sre_constants import SUCCESS
from statistics import mode
import cv2
from scipy.misc import face
from train import faceTrain

def main():
    model = faceTrain()
    model.load_model("trained")
    print("starting Video...Please Make sure Webcam is enabled")
    cap=cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    cap.set(10,180)
    while True:
        success, img = cap.read()
        cv2.imshow("video",img)
        if cv2.waitKey(1) == 113:
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()
