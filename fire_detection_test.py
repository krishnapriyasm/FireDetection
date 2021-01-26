   
import cv2
import numpy as np
import time
from PIL import Image

Fire_Reported = 0
xyz = 0



cap = cv2.VideoCapture("forest5.avi") # If you want to use webcam use Index like 0,1.
print("video read successssssss","\n")
fps    = int(cap.get(cv2.CAP_PROP_FPS))
print("FPS is " ,fps)
length = cap.get(7)
print("Frame count is " , length )

while True:
    (grabbed, frame) = cap.read()
    if not grabbed:
        break

    frame = cv2.resize(frame, (320, 240)) 
 
    blur = cv2.GaussianBlur(frame, (21, 21), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
   #Image.fromarray(hsv).save('hsv_frame_testing.jpg')

    lower = [0, 10, 165]
    upper = [30, 255, 255]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    mask = cv2.inRange(hsv, lower, upper)
 
    output = cv2.bitwise_and(frame, hsv, mask=mask)
    
    no_of_red = cv2.countNonZero(mask)


    
    
    if int(no_of_red) > 1000:
        Fire_Reported = Fire_Reported + 1


    '''_,conts,h=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame,conts,-1,(255,0,0),3)
    length= length+1
    for i in range(len(conts)):
    	x,y,w,h=cv2.boundingRect(conts[i])
    	roi=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    	roi=frame[y:y+h,x:x+w]
    	area = roi.size


    if area>50000:
        if xyz<3:
            print("object detectes")
            print (length)
            xyz=xyz +1'''
    
    
    time.sleep(1/fps)
    cv2.imshow("input", frame)
    cv2.imshow("output", output)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Fire_Reported in  " , Fire_Reported , " no. of frames")
print("last line")
 
cv2.destroyAllWindows()
cap.release()
