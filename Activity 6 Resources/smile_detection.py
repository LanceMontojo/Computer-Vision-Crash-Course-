import cv2

def detect():
  
  smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
  camera = cv2.VideoCapture(0)

  while True:
    ret, frame = camera.read()
    if not ret:
      break
      
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smiles = smile_cascade.detectMultiScale(gray, 1.8, 20)

    for (sx, sy, sw, sh) in smiles:
      cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    cv2.imshow("camera", frame)
    
    key = cv2.waitKey(int(1000 / 12)) & 0xff
    if key == ord("q") or key == 27:  
      break

  camera.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  detect()
