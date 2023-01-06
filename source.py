# import modules
import cv2
from train import faceTrain

def main():
    model = faceTrain()
    print("starting Video...Please Make sure Webcam is enabled")
    cap=cv2.VideoCapture(0)
    faceCascade = cv2.CascadeClassifier("./pretrained/frontalFace.xml")
    cap.set(3,640)
    cap.set(4,480)
    cap.set(10,180)
    while True:
        success, img = cap.read()
        img = cv2.flip(img,1)
        grayScaledImage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = faceCascade.detectMultiScale(
            grayScaledImage,
            scaleFactor=1.1,
            minSize=(96, 96),
        )
        croppedFaces = []
        for (x, y, w, h) in faces:
            croppedFaces.append(cv2.resize(grayScaledImage[y:y+h, x:x+w],(96, 96)));
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 127), 2)
        for imageIndex in range(len(croppedFaces)):
            result = model.predict(croppedFaces[imageIndex], "trained")
            for point in result:
                cv2.circle(croppedFaces[imageIndex], [point[0], point[1]], 1, (255, 0, 0), 2)

        [cv2.imshow(str(imgIndex), croppedFaces[imgIndex]) for imgIndex in range(len(croppedFaces))]
        cv2.imshow("Output", img)
        if cv2.waitKey(1) == 113:
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()
