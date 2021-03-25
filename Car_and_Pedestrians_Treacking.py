import cv2

#local image
#img_file = 'carImage.jpg'
video = cv2.VideoCapture('dataset_video1.avi')
#this file not have people, use vdio with people

#pre-trained car classifier
classifier_file = 'cars.xml'

#pre-trained human classifier
classifier_file_human = 'human.xml'

#create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

#create human classifier
human_tracker = cv2.CascadeClassifier(classifier_file_human)

#run untill
while True:

    #read current frame from vedio
    (read_succesful, frame) = video.read()

    if read_succesful:
        #convert frame to grayscale
        grayscaled_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect car coordinates
    cars = car_tracker.detectMultiScale(grayscaled_frame)

    # detect human coordinates
    humans = human_tracker.detectMultiScale(grayscaled_frame)

    # draw rectangle around the image based on coordinates
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # draw rectangle around the image(human) based on coordinates
    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # display image with cars detected
    cv2.imshow('Ghost car detecter', frame)

    # wait till key pressed
    key = cv2.waitKey(1)

    #stop if Q is pressed
    if key==81 or key==113:
        break

#release vedio
video.release()

""""
#creating opencv image
img = cv2.imread(img_file)

    #convert img to grayscale
    blck_n_white = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #create car classifier
    car_tracker = cv2.CascadeClassifier(classifier_file)

    #detect car coordinates
    cars = car_tracker.detectMultiScale(blck_n_white)

    #draw rectangle around the image based on coordinates
    for(x,y,w,h) in cars:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #display image with cars detected
    cv2.imshow('Ghost car detecter',img)

#wait till key pressed
cv2.waitKey()
"""

print("code completed!")