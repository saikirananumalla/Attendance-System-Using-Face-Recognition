#from django.shortcuts import render

# Create your views here.
#from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from .forms import ImageForm,UrlForm
from .models import Image
from werew.settings import BASE_DIR
from django.http import HttpResponse,HttpResponseForbidden
import os
import datetime
from datetime import datetime as dt
from datetime import timedelta as td
#from datetime import date
import pandas as pd
import pickle as pkl
#ML
#from sklearn import svm
import numpy
#import PIL
import cv2
import sys
#import matplotlib.pyplot as plt
#import numpy
#import scipy.misc
#from collections import namedtuple
#from PIL import Image as Img
#ML

def home(request):
    #print(new_folder)
    return render(request, "index.html")
def loadcsv(request):
    return render(request, "myTable.htm")
def trainImage(request):
    f=open(os.path.join(BASE_DIR,'a'),'rb')
    a=pkl.load(f)
    f.close()
    #print(a)
    if request.method=="POST":
        form = ImageForm(request.POST,request.FILES)
        if form.is_valid():
            nname=form.save()
            #print(nname)
            size = 4
            fn_haar = 'haarcascade_frontalface_default.xml'
            fn_dir = os.path.join(BASE_DIR,'faceio')
            try:
                fn_name = nname.name
            except:
                print("You must provide a name")
                sys.exit(0)
            path = os.path.join(fn_dir, str(fn_name))
            if not os.path.isdir(path):
                os.mkdir(path)
                print(path)
                a.append(path)
                #print(a)
                f=open(os.path.join(BASE_DIR,'a'),'wb')
                t=pkl.dump(a,f)
                f.close()
                print(t)
            (im_width, im_height) = (112, 92)
            haar_cascade = cv2.CascadeClassifier(fn_haar)
            webcam = cv2.VideoCapture(0)

            # Generate name for image file
            pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
                if n[0]!='.' ]+[0])[-1] + 1
            #print(pin)
            # Beginning message
            print("\n\033[94mThe program will save 20 samples. \
            Move your head around to increase while it runs.\033[0m\n")

            # The program loops until it has 20 images of the face.
            count = 0

            pause = 0
            count_max = 20
            while count < count_max:

                # Loop until the camera is working
                rval = False
                while(not rval):
                    # Put the image from the webcam into 'frame'
                    (rval, frame) = webcam.read()
                    if(not rval):
                        print("Failed to open webcam. Trying again...")

                # Get image size
                height, width, channels = frame.shape

                # Flip frame
                frame = cv2.flip(frame, 1, 0)

                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Scale down for speed
                mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

                # Detect faces
                faces = haar_cascade.detectMultiScale(mini)

                # We only consider largest face
                faces = sorted(faces, key=lambda x: x[3])
                if faces:
                    face_i = faces[0]
                    (x, y, w, h) = [v * size for v in face_i]

                    face = gray[y:y + h, x:x + w]
                    face_resize = cv2.resize(face, (im_width, im_height))

                    # Draw rectangle and write name
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(frame, fn_name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,
                        1,(0, 255, 0))

                    # Remove false positives
                    if(w * 6 < width or h * 6 < height):
                        print("Face too small")
                    else:

                        # To create diversity, only save every fith detected image
                        if(pause == 0):

                            print("Saving training sample "+str(count+1)+"/"+str(count_max))

                            # Save image file
                            cv2.imwrite('%s/%s.png' % (path, pin), face_resize)

                            pin += 1
                            count += 1

                            pause = 1

                if(pause > 0):
                    pause = (pause + 1) % 5
                cv2.imshow('Training', frame)
                key = cv2.waitKey(10)
                if count == 19:
                    webcam.release()
                    cv2.destroyAllWindows()
                    return render(request, "tsuccess.html")
            return render(request, "train.html", {'form':form,'nname':nname},)
    else:
        form = ImageForm()

    return render(request, "train.html",{'form':form,})
#print(new_folder)
def testImage(request):
    prediction=[1,2]
    if request.method=="POST":
        form = UrlForm(request.POST,request.FILES)
        if form.is_valid():
            count=0
            fsave=form.save()
            size = 1
            fn_haar = 'haarcascade_frontalface_default.xml'
            fn_dir = os.path.join(BASE_DIR,'faceio')

            # Part 1: Create fisherRecognizer
            print('Training...')

            # Create a list of images and a list of corresponding names
            (images, lables, names, id) = ([], [], {}, 0)
            (newimage,newlabel)=([],[])
            # Get the folders containing the training data
            for (subdirs, dirs, files) in os.walk(fn_dir):

                # Loop through each folder named after the subject in the photos
                for subdir in dirs:
                    names[id] = subdir
                    subjectpath = os.path.join(fn_dir, subdir)

                    # Loop through each photo in the folder
                    for filename in os.listdir(subjectpath):

                        # Skip non-image formates
                        f_name, f_extension = os.path.splitext(filename)
                        if(f_extension.lower() not in
                            ['.png','.jpg','.jpeg','.gif','.pgm']):
                            print("Skipping "+filename+", wrong file type")
                            continue
                        path = subjectpath + '/' + filename
                        lable = id
                        if subdir==a[-1]:
                            newimage.append(cv2.imread(path,0))
                            newlabel.append(int(lable))
                        # Add to training data
                        images.append(cv2.imread(path, 0))
                        lables.append(int(lable))
                    id += 1
            (im_width, im_height) = (112, 92)

            # Create a Numpy array from the two lists above
            (images, lables) = [numpy.array(lis) for lis in [images, lables]]
            #(newimage, newlable) = [numpy.array(lis) for lis in [newimage, newlable]]
            # OpenCV trains a model from the images
            # NOTE FOR OpenCV2: remove '.face'
            ca=pd.read_csv('att.csv')
            ca.set_index('Name',inplace=True)
            prd=str(datetime.date.today())
            if prd not in ca.keys():
                ca[prd]=0
            model = cv2.face.LBPHFaceRecognizer_create()
            #model.train(images, lables)
            model.read('saiki.yml')



            # Part 2: Use fisherRecognizer on camera stream
            haar_cascade = cv2.CascadeClassifier(fn_haar)
            webcam = cv2.VideoCapture(0)
            i=0
            pdo=dt.now()
            ed=pdo+td(seconds=20)
            cnt=0
            while True:

                # Loop until the camera is working
                rval = False
                while(not rval):
                    # Put the image from the webcam into 'frame'
                    (rval, frame) = webcam.read()
                    if(not rval):
                        print("Failed to open webcam. Trying again...")

                # Flip the image
                frame=cv2.flip(frame,1,0)

                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Resize to speed up detection (optinal, change size above)
                mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

                # Detect faces and loop through each one
                faces = haar_cascade.detectMultiScale(mini)
                for i in range(len(faces)):
                    face_i = faces[i]

                    # Coordinates of face after scaling back by 'size'
                    (x, y, w, h) = [v * size for v in face_i]
                    face = gray[y:y + h, x:x + w]
                    face_resize = cv2.resize(face, (im_width, im_height))

                    # Try to recognize the face
                    prediction = model.predict(face_resize)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255),5)

                    # [1]
                    # Write the name of recognized face ,prediction[1]/10,-%.0f
                    cv2.putText(frame,
                    '%s' % (names[prediction[0]]),
                    (x-5, y-5), cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0))
                    i=i+1
                # Show the image and check for ESC being pressed
                cv2.imshow('expFD', frame)
                key = cv2.waitKey(10)
                #if prediction[0]==36:
                    #webcam.release()
                    #cv2.destroyAllWindows()
                    #break
                #print(names[prediction[0]])
                #cv2.destroyAllWindows()
                if names[prediction[0]]==fsave.name:
                    if cnt<10:
                        cnt+=1
                    else:

                        print(prd)
                        ca[prd][names[prediction[0]]]=1
                        ca.to_csv('att.csv')
                        df = pd.read_csv('att.csv')
                        df.to_html('templates/myTable.htm')
                        webcam.release()
                        cv2.destroyAllWindows()
                        break
                if dt.now()>ed:
                    webcam.release()
                    cv2.destroyAllWindows()
                    return render (request,"fail.html",{'fsave': fsave,'form':form},)
            return render(request, "success.html", {'fsave': fsave,'form':form},)
    else:
        form = UrlForm()
    return render(request, "test.html",{'form':form})
def AdminTrain(request):
    prediction=[1,2]
    if request.method=="POST":
        form = UrlForm(request.POST,request.FILES)
        if form.is_valid():
            f=open(os.path.join(BASE_DIR,'a'),'rb')
            aval=pkl.load(f)
            f.close()
            fsave=form.save()
            if fsave.name!='anumalla':
                return render(request,'adfail.html')
            size = 1
            fn_haar = 'haarcascade_frontalface_default.xml'
            fn_dir = os.path.join(BASE_DIR,'faceio')

            # Part 1: Create fisherRecognizer
            print('Training...')

            # Create a list of images and a list of corresponding names
            (images, lables, names, id) = ([], [], {}, 0)
            #(newimage,newlabel)=([],[])
            # Get the folders containing the training data
            for (subdirs, dirs, files) in os.walk(fn_dir):

                # Loop through each folder named after the subject in the photos
                for subdir in dirs:
                    names[id] = subdir
                    subjectpath = os.path.join(fn_dir, subdir)

                    # Loop through each photo in the folder
                    for filename in os.listdir(subjectpath):

                        # Skip non-image formates
                        f_name, f_extension = os.path.splitext(filename)
                        if(f_extension.lower() not in
                            ['.png','.jpg','.jpeg','.gif','.pgm']):
                            print("Skipping "+filename+", wrong file type")
                            continue
                        path = subjectpath + '/' + filename
                        lable = id
                        '''try:
                            if subdir==aval[-1]:
                                newimage.append(cv2.imread(path,0))
                                newlabel.append(int(lable))
                        except:
                            return render(request,"index.html")'''
                        # Add to training data
                        images.append(cv2.imread(path, 0))
                        lables.append(int(lable))
                    id += 1
            (im_width, im_height) = (112, 92)

            # Create a Numpy array from the two lists above
            (images, lables) = [numpy.array(lis) for lis in [images, lables]]
            #(newimage, newlabel) = [numpy.array(lis) for lis in [newimage, newlabel]]
            model = cv2.face.LBPHFaceRecognizer_create()
            model.train(images, lables)
            #model.update(newimage,newlabel)
            model.save('saiki.yml')
            return render(request, "adsu.html", {'fsave': fsave,'form':form},)
    else:
        form = UrlForm()
    return render(request, "admin.html",{'form':form,},)
