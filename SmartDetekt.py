import requests
import json
import urllib
import cv2,os
import numpy as np
import time

from PIL import Image



class EmployeeData:
    eid = 0
    name = ''
    absentStart = 0
    absentEnd = 0
    absentDuration = 0
    present = 0
    
    def __init__(self, myid, myname):
        self.eid = myid
        self.name = myname

    def getEmpId(self):
        return self.eid

    def getEmpName(self):
        return self.name

    

serverUrl = 'http://18.188.118.204:3000/api'
projFolder = '/home/pi/detectioncode/'

shopid = 1000

throsholdAbsent = 200     #In Seconds

starttime=time.time()

shopdetails={}

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier(projFolder+'haarcascade_frontalface_default.xml');
countertime = 0

empList = {}

def getDataFromREST():
    global shopdetails
    global admin_id
    r=requests.get(serverUrl+'/shop/'+str(shopid))
    shopdetails=r.json()
    admin_id=shopdetails['supervisoradminid']
    uri = serverUrl+'/employes?filter=%7B%22where%22%3A%20%7B%22shopid%22%3A'+ str(shopid) +'%7D%7D'
    header ='Content-Type: application/json';
    r = requests.get(uri)
    employees = r.json()

    for emp in employees:
        empid = emp['id']
        empname = emp['name']
        oneEmp = EmployeeData(empid, empname)
        empList[empid] = oneEmp
        print(str(empid))
        filelisturl = serverUrl+'/allfiles/'+str(empid)+'/files'
        r1 = requests.get(filelisturl)
        container=r1.json()
        for employ in container:
            fileurl = serverUrl+'/allfiles/'+str(empid)+'/download/'+employ['name']
            downloadFile(fileurl)


def downloadFile(url):
    file_name = url.split('/')[-1]
    file_name = file_name.replace('..', '_')
    u = urllib.request.urlopen(url)
    flagpresent=0
    for file in os.listdir(projFolder+'dataSet/'):
       if file== file_name:
           flagpresent=1
           break
    if flagpresent ==0:    
        f = open( projFolder+'dataSet/'+file_name, 'wb')
        meta = u.info()
        file_size = meta.get("Content-Length")
        blocksz = 8192
        file_size_new = 0
        while True:
            buffer = u.read(blocksz)
            if not buffer:
                break
            file_size_new += len(buffer)
            f.write(buffer)

        f.close()


def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #create empth face list
    faceSamples=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:

        # Updates in Code
        # ignore if the file does not have jpg extension :
        if(os.path.split(imagePath)[-1].split(".")[-1]!='jpg'):
            continue

        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces=detector.detectMultiScale(imageNp)
        #If a face is there then append that in the list as well as Id of it
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples,Ids

def runTrainer():  
   faces,Ids = getImagesAndLabels(projFolder+'dataSet')
   recognizer.train(faces, np.array(Ids))
   recognizer.write(projFolder+'trainner/trainner.yml')


def runDetector():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(projFolder+'trainner/trainner.yml')
    cascadePath = projFolder+"haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);
    font = cv2.FONT_HERSHEY_SIMPLEX
    #iniciate id counter
    id = 0
    
    # names related to ids: example ==> Marcelo: id=1,  etc
    # names = ['None', 'parkavi', 'roopa','pooja','adfgh','tohhh'] 

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height

    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
        
    while True:
       global countertime
       countertime+=1
       ret,img =cam.read()
       #img = cv2.flip(img, -1) # Flip vertically

       gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

       faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

       for(x,y,w,h) in faces:

          cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

          id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
          print(id)
          # Check if confidence is less them 100 ==> "0" is perfect match 
          if (confidence < 100):
              if(id in empList.keys()):
                 detdEmp = empList[id]
                 detdname = detdEmp.getEmpName()
                 detdEmp.present += 1
                 empList[id] = detdEmp
                 confidence = "  {0}%".format(round(100 - confidence))
          else:
              detdname = "unknown"
              confidence = "  {0}%".format(round(100 - confidence))
              
        
          cv2.putText(img, str(detdname), (x+5,y-5), font, 1, (255,255,255), 2)
          cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
          for key, myemp in empList.items():
              print('Id: ' + str(key) + ' Present' + str(empList[key].present) + ' global counter '+ str(countertime) )
          
          if(countertime > throsholdAbsent):
              #send alert to Administrator, check if the employee is present atleast half the time
              halfcntr = countertime/2
              countertime = 0   #reset counter
              for key, myemp in empList.items():
                  if(halfcntr > empList[key].present):         #Employee is not detected for half the time
                      r=requests.get(serverUrl+'/administrators/'+str(admin_id))
                      contactno=r.json()
                      cno=contactno['contactnum']
                      global starttime
                      global shopdetails
                      uri = serverUrl+'/absencenotifications'
                      header ='Content-Type: application/json';
                      mins=throsholdAbsent/60
                      data = {'empid':key, 'shopid':shopid, 'location': shopdetails['address'],  'absentstarttime':time.localtime(starttime),
                      'absentendtime':time.localtime(time.time()), 'durationabsent':mins}
                      r = requests.post(url = uri, data = data)
                      print(r.json())
                      smsuri = ('http://smst.duratechsolutions.in/api/schedulemsg.php?user=duratechit&pass=duratechit&sender=DURATE&phone=7200845985&'+
                      'text=Employee ID '+str(empList[key]) + ' at Shop ID '+ str(shopid) + 'is not present in shop for last '+str(mins)+' minutes'+
                      '&priority=ndnd&stype=normal');
                      detdEmp = empList[key]
                      detdname = detdEmp.getEmpName()
                      smsuri = 'http://smst.duratechsolutions.in/api/sendmsg.php?user=duratechit&pass=duratechit&sender=DURATE&phone='+str(cno)+'&text= Employee '+str(detdname)+' ID '+str(key)+'AbsentAtShop'+str(shopid)+'for'+str(mins)+'&priority=ndnd&stype=normal'
                      r = requests.post(url = smsuri)
                      starttime=time.time()
              empList[key].present = 0

       cv2.imshow('camera',img)

       k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
       if k == 27:
           break
       

        # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()


#def doInBackground():
    #Check if all the employees are present

        

getDataFromREST()
runTrainer()
runDetector()



