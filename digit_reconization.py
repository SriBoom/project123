import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

#Google Colab is available on the browser. When we go to any link, many of the links start with an https:// instead of an http:// The s in the https:// means that we are trying to establish a secure connection. Browser provides us with SSL when we try to open a link, but here since we are running a python script locally on our machine, the openML thinks that our python script cannot be trusted since there is no SSL, which a browser automatically provides. To encounter this, if we look at the code above, we imported a library known as ssl
#Here, we first have an if condition where we are checking if our python environment’s “PYTHONHTTPSVERIFY” is an empty string. This means that our Python script is not HTTPS Verified. Next, we are seeing if our ssl’s unverified context is created or not. Remember, a browser provides the ssl context by default. Now, if our script does not have https verification and if it also does not have an ssl’s unverified context, then we are creating a default https unverified context for our python script using the SSL module.
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and 
    #when you connect with the web browser, it will defaultly provide ssl content which is secured. 
    getattr(ssl, '_create_unverified_context', None)): 
    #if you don't have https verfication(line 15), we will be creating default_https_context. 
    ssl._create_default_https_context = ssl._create_unverified_context

#fetching datasets of handwriten digits. X is the data of the image(represetnted in binary) and y is the label of the image
X,y = fetch_openml('mnist_784', version=1, return_X_y=True)
X= np.array(X)
print(pd.Series(y).value_counts())
classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
nclasses=len(classes)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=9, train_size=7500, test_size=2500)
X_train_scaled=X_train/255.0
X_test_scaled=X_test/255.0

clf = LogisticRegression(solver="saga", multi_class="multinomial").fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

#Starting the camera
cap = cv2.VideoCapture(0)

while(True):
  # Capture frame-by-frame
  try:
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Drawing a box in the center of the video
    height, width = gray.shape
    upper_left = (int(width / 2 - 56), int(height / 2 - 56))
    bottom_right = (int(width / 2 + 56), int(height / 2 + 56))
    cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

    #To only consider the area inside the box for detecting the digit
    #roi = Region Of Interest
    roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

    #Converting cv2 image to pil format
    im_pil = Image.fromarray(roi)

    # convert to grayscale image - 'L' format means each pixel is 
    # represented by a single value from 0 to 255
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)

    image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized_inverted)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_pred = clf.predict(test_sample)
    print("Predicted class is: ", test_pred)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  except Exception as e:
    pass

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()