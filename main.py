from django import conf
import numpy as np
import cv2

"""
info extraida de https://pjreddie.com/darknet/yolo/
"""


#load yolo

cap = cv2.VideoCapture(0)

whT = 320

confidenceThreshold = 0.5
nmsThresholds = 0.4

classesFile = 'coco.names'

classNames = []


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfiguration = 'yolov3.cfg'
modelweights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelweights)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    hT, wT, cT = img.shape

    bbox = []

    classIds = []

    confs = []

    for output in outputs:
        for det in output:
          scores = det[5:]
          classId = np.argmax(scores)
          confidence = scores[classId]
          if confidence > confidenceThreshold:
            w, h = int(det[2]*wT) , int(det[3]*hT)
            x, y = int((det[0]*wT) - w/2),int((det[1]*wT) - w/2)
            bbox.append([x,y,w,h])
            classIds.append(classId)
            confs.append(float(confidence))
    
    indices = cv2.dnn.NMSBoxes(bbox,confs,confidenceThreshold,nmsThresholds)


    for i in indices:
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]

        if classNames[classIds[i]] == "person":
            color = (0,255,0)

        elif classNames[classIds[i]] == "car":
            color = (0,0,255)
            
        else:
            color = (255,0,255)
        
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%', 
            (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
        
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------        
            
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

while True:

    succes,img = cap.read()

    blob = cv2.dnn.blobFromImage(img,1/125,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()

    outputNames = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)
    
    findObjects(outputs,img)

    #img = ResizeWithAspectRatio(img, width=1000) # Ajustamos manteniendo un ancho de 1024
    
    cv2.imshow('Detector de personas',img)

    if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 






        
    
cap.release()
cv2.destroyAllWindows()