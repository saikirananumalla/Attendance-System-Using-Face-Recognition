# exploratory-project-sem4
“Attendance System using Face Recognition”  is an automated attendance management system model developed to tackle the challenges with existing attendance system such as wastage of time during lectures , mis-marking of attendance, proxy etc… 

Open-cv’s inbuilt cascades like haar_cascade frontal face detector are employed to identify faces from the images ,this serves the face detection part. For face recognition, LBP algorithm is implemented using the lbph face recognizer, which is the best among other opencv face recognizers available.

Finally, a web-app was created with an interference to register and to mark their attendance using webcam capture for localhost and image upload for online webapp .   

To run the project 
pip install opencv-contrib-python
pip install Pillow
pip install pandas
pip install django==2.1.5
python manage.py makemigrations
python manage.py migrate
python manage.py runserver

Working demo can be found here : https://www.youtube.com/watch?v=mZftbFmFAd4
Webapp based on this can be found here: https://saiky.pythonanywhere.com
