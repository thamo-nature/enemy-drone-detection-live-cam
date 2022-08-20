from keras.models import load_model
import keras
print("keras version " + keras.__version__)

import numpy
print("numpy version " + numpy.__version__)

import tensorflow
print("tensorflow version " + tensorflow.__version__)

import h5py
print("h5py version " + h5py.__version__)

from PIL import Image, ImageOps
import numpy as np

import cv2
print("cv2 version " + cv2.__version__)

import os
import smtplib
import time
import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

#import RPi.GPIO as GPIO
from time import sleep

# GPIO.setwarnings(False)
# GPIO.setmode(GPIO.BCM)
# 
# GPIO.setup(17,GPIO.OUT) #GPIO 4 -> Motor 1 terminal A

def stop():
        #GPIO.output(17,True) #2A+
        pass


strFrom = 'from@gmail.com'
strTo = 'to@gmail.com'


def send_mail(drone_type):

    # Define these once; use them twice!
    DateTimefilename = datetime.datetime.now() .strftime ("%Y-%m-%d-%H.%M.%S")
    
    # Create the root message and fill in the from, to, and subject headers
    msgRoot = MIMEMultipart('related')
    msgRoot['Subject'] = 'Waste Alert!!! ' + DateTimefilename
    msgRoot['From'] = strFrom
    msgRoot['To'] = strTo
    
    msgRoot.preamble = 'This is a multi-part message in MIME format.'

    # Encapsulate the plain and HTML versions of the message body in an
    # 'alternative' part, so message agents can decide which they want to display.
    msgAlternative = MIMEMultipart('alternative')
    msgRoot.attach(msgAlternative)


    msgText = MIMEText('This is the alternative plain text message.')
    msgAlternative.attach(msgText)

    # We reference the image in the IMG SRC attribute by the ID we give it below
#     msgText = MIMEText('<b>Garbage <i>Waste</i> - </b> Detected.<br><img src="cid:image1"><br> From: Alpha Technologies!', 'html')
#     msgAlternative.attach(msgText)

    if(drone_type == 0):
            # We reference the image in the IMG SRC attribute by the ID we give it below
            msgText = MIMEText('<b>Enemy Drone <i>White</i> - </b> Detected.<br><img src="cid:image1"><br> From: Alpha Technologies!', 'html')
            msgAlternative.attach(msgText)
            # This example assumes the image is in the current directory
            fp = open('/root/projects/drone_detection/white_enemy/0.jpg', 'rb')
    elif(drone_type == 1):
            # We reference the image in the IMG SRC attribute by the ID we give it below
            msgText = MIMEText('<b>Friend Drone <i>Black</i> - </b> Detected.<br><img src="cid:image1"><br> From: Alpha Technologies!', 'html')
            msgAlternative.attach(msgText)
            # This example assumes the image is in the current directory
            fp = open('/root/projects/drone_detection/black_friend/0.jpg', 'rb')
            
    msgImage = MIMEImage(fp.read())
    fp.close()

    # Define the image's ID as referenced above
    msgImage.add_header('Content-ID', '<image1>')
    msgRoot.attach(msgImage)
    
    # Send the email (this example assumes SMTP authentication is required)
    
    smtp = smtplib.SMTP('smtp.gmail.com', 587)
    smtp.starttls()
    smtp.login( strFrom, 'set_mail_password_here')    
    smtp.sendmail(strFrom, strTo, msgRoot.as_string())
    smtp.quit()
    print ("Thank You E-Mail has been Send to %s" % strTo)


video = cv2.VideoCapture(0)

# Load the model
model = load_model('keras_model.h5')

frame_count = 10

finalize_white_enemy_drone_count = []
finalize_black_friend_drone_count = []

while True:
    
        _, frame = video.read()
                
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        im = Image.fromarray(color_coverted)

        size = (224, 224)
        
        image = ImageOps.fit(im, size, Image.ANTIALIAS)        

        image_array = np.asarray(image)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data)
        class_name = ['white_friend','black_enemy','no_wastes']
        output = [class_name[np.argmax(prediction)]]
        #print(output)
        #print([class_name[np.argmax(prediction)]])
#        print(prediction)

#         d_w = np.argmax(prediction)
#         print(d_w)
        
        x=prediction[0][0]
        y=prediction[0][1]
        #z=prediction[0][2]
        
        print("X is " + str(x))
        print("Y is " + str(y))
#         print(z)

        if(x > 0.9 and y < 0.5):
            print("Found White Enemy Drone")            
            finalize_white_enemy_drone_count.append("white_friend")
            #print(finalize_white_enemy_drone_count)
            #process last 10 frames
            if( len(finalize_white_enemy_drone_count) >= 10 ):
               #check if the list has all 10 frames same obj 
               if(all(map(lambda x: x == finalize_white_enemy_drone_count[0], finalize_white_enemy_drone_count[1:]))):
                   print("Last" + str(frame_count) + "Frame Accuracy White Friend Drone")
                   finalize_white_enemy_drone_count.clear()
                   send_mail(0)
               #stop()
               #sleep(15)
               
        elif(x < 0.5 and y > 0.9):
            print("Found Black Friend Drone")            
            finalize_black_friend_drone_count.append("white_friend")
            #print(finalize_white_enemy_drone_count)
            #process last 10 frames
            if( len(finalize_black_friend_drone_count) >= 10 ):
               #check if the list has all 10 frames same obj 
               if(all(map(lambda x: x == finalize_black_friend_drone_count[0], finalize_black_friend_drone_count[1:]))):
                   print("Last" + str(frame_count) + "Frame Accuracy Black Enemy Drone")
                   finalize_black_friend_drone_count.clear()
                   send_mail(1)
               #stop()
               #sleep(15)
        
        else:
            print("No Drone Found")


        cv2.imshow("Prediction", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break

video.release()
cv2.destroyAllWindows()

