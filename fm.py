import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.fft import fft,fftfreq


cap = cv2.VideoCapture('barbeq.avi')
frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

#fourcc = cv2.VideoWriter_fourcc('X','V','I','D')

#out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280,720))

print("video read successsssss","\n")
fps    = int(cap.get(cv2.CAP_PROP_FPS))
print("FPS is " ,fps)
length = cap.get(7)
print("Frame count is " , length )


ret2, frame2 = cap.read()


#print(frame1.shape)
area_array = []
frame_number_array = []
i = 0
while True:

    i = i+1

    ret1, frame1 = ret2, frame2
    ret2, frame2 = cap.read()
    if frame2 is None:
        break
    
    lower = [0, 10, 165]
    upper = [30, 255, 255]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    blur_1 = cv2.GaussianBlur(frame1, (21, 21), 0)
    blur_2 = cv2.GaussianBlur(frame2, (21, 21), 0)

    hsv_1 = cv2.cvtColor(blur_1, cv2.COLOR_BGR2HSV)
    hsv_2 = cv2.cvtColor(blur_2, cv2.COLOR_BGR2HSV)

    mask_1 = cv2.inRange(hsv_1, lower, upper)
    mask_2 = cv2.inRange(hsv_2, lower, upper)
    
    output_1 = cv2.bitwise_and(frame1, hsv_1, mask=mask_1)
    output_2 = cv2.bitwise_and(frame2, hsv_2, mask=mask_2)
    
    diff = cv2.absdiff(output_1, output_2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(gray, 20, 180, cv2.THRESH_BINARY)


   
    
    
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

   # print("contours :")
   # print(contours)

    contour_array = []

    img = cv2.drawContours(diff, contours, -1, (0,255,0), 3)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 500:
            continue
        contour_array.append( cv2.contourArea(contour) )
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(diff, (x, y), (x+w, y+h), (0, 255, 0), 2)


        #cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 3)
    #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2) #small contours

    if  len(contour_array) != 0:
        #print(contour_array)
        area_array.append(  max(contour_array)  )
        frame_number_array.append(i)
    else :
        area_array.append(  0  )
        frame_number_array.append(i)
    print(i)

    # time.sleep(1/fps)
    
    image = cv2.resize(frame1, (1280,720))
    #out.write(image)
    cv2.imshow("feed", frame1)
    cv2.imshow("img", img)
    cv2.imshow("diff_output", diff)
    cv2.imshow("output", output_1)

    # frame1 = frame2
    # ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

print(area_array)
area_array.pop

plt.plot(frame_number_array,area_array)
plt.ylabel('Area')
plt.show()
yf = fft(area_array)
xf = fftfreq(int(length-1), 1)

plt.plot(xf, np.abs(yf))
plt.show()


cv2.destroyAllWindows()
cap.release()
#out.release()
