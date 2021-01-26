import cv2
import numpy as np
import time

cap = cv2.VideoCapture('fire1.avi')
frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

#fourcc = cv2.VideoWriter_fourcc('X','V','I','D')

#out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280,720))

print("video read successssssss","\n")
fps    = int(cap.get(cv2.CAP_PROP_FPS))
print("FPS is " ,fps)
length = cap.get(7)
print("Frame count is " , length )




#print(frame1.shape)
while True:

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    if frame1 is None:
        break
    
    lower = [0, 10, 165]
    upper = [30, 255, 255]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    blur_1 = cv2.GaussianBlur(frame1, (21, 21), 0)
    hsv = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    
    output = cv2.bitwise_and(diff, hsv, mask=mask)
    
    
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 900:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 3)
    #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2) #small contours


    time.sleep(1/fps)
    
    image = cv2.resize(frame1, (1280,720))
    #out.write(image)
    cv2.imshow("feed", frame1)
    cv2.imshow("output", output)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
#out.release()
