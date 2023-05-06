# importing
import cv2

# Storing in the video
cap = cv2.VideoCapture('walking.avi')

# Taking the dectector
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_classifier.detectMultiScale(gray, 1.9, 1)
    
    # Display the resulting frame
    for (x,y,w,h) in bodies:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
         
         
    #displaying
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == 32:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
