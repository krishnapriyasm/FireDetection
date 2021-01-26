import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.fft import fft,fftfreq


cap = cv2.VideoCapture('forest1.avi')
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

contour_fire_pixel_colors_means_array = []
fire_intensity_trend_array = []
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
    diff_freq = diff
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(gray, 20, 180, cv2.THRESH_BINARY)


   
    
    
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

   # print("contours :")
   # print(contours)

    contour_array = []

    img = cv2.drawContours(diff, contours, -1, (0,255,0), 3)

    biggest_area =0
    biggest_contour = 0

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)


        if  int(cv2.contourArea(contour)) > biggest_area  :
            biggest_area = int(cv2.contourArea(contour))
            biggest_contour = contour


        if cv2.contourArea(contour) < 500:
            continue


        contour_array.append( cv2.contourArea(contour) )
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(diff, (x, y), (x+w, y+h), (0, 255, 0), 2)


        #cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 3)
    #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2) #small contours

    if  len(contours) != 0:
        #print(contour_array)
        # area_array.append(  max(contour_array)  )
        # frame_number_array.append(i)
        contour_fire_pixel_colors = []

        hsv_pic  = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        frame_number_array.append(i)

        for z in biggest_contour : # For each coordinate in the biggest contour add to contour_fire_pixel_colors array.
            x,y = z[0] # z = [[22 44]], z[0] = [22 44] tuples, sliceing of aarays


            color = hsv_pic[y, x]
            contour_fire_pixel_colors.append(color[2])# Value / Brigness part TODO: Make it better maybe a combo of V and S
            # print(color)
        contour_fire_pixel_colors_means_array.append( np.mean(contour_fire_pixel_colors) )
        print( "i : ",i,"  mean: ", np.mean(contour_fire_pixel_colors))

        if i == 1 :
            fire_intensity_trend_array.append(0)
        else :
            if abs(contour_fire_pixel_colors_means_array[i-1] - contour_fire_pixel_colors_means_array[i-2]) < 0.0001 :
                fire_intensity_trend_array.append(0)
            elif contour_fire_pixel_colors_means_array[i-1] > contour_fire_pixel_colors_means_array[i-2] :
                fire_intensity_trend_array.append(1)
            else :
                fire_intensity_trend_array.append(-1)
                

            

        freq_img = cv2.drawContours(frame2,biggest_contour, -1, (0,255,0), 3)

    else :
        freq_img = frame1
        # area_array.append(  0  )
        contour_fire_pixel_colors_means_array.append( 0 )
        fire_intensity_trend_array.append(0)
        
        frame_number_array.append(i)
    # print(i)

    # time.sleep(1/fps)
    
    image = cv2.resize(frame1, (1280,720))
    #out.write(image)
    cv2.imshow("feed", frame1)
    cv2.imshow("img", img)
    cv2.imshow("diff_output", diff)
    cv2.imshow("output", output_1)
    cv2.imshow("freq_image", freq_img)


    # frame1 = frame2
    # ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

# print(area_array)
# area_array.pop

# s=plt.plot(frame_number_array,contour_fire_pixel_colors_means_array)
s=plt.plot(frame_number_array,fire_intensity_trend_array)
plt.ylabel('Avg Color')
plt.show()
yf = fft(fire_intensity_trend_array)
xf = fftfreq(int(length-1), 1)

plt.plot(xf, np.abs(yf))
plt.show()


cv2.destroyAllWindows()
cap.release()
#out.release()
