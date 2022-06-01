import cv2

thres = 0.55  # Threshold to detect object

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

classNames = []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('n').split('n')

classNames = ['People','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
              'traffic light','fire hydrant','street sign','stop sign','parking meter','bench','bird','cat',
              'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','hat','backpack','umbrella','shoe',
              'eye glasses','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat',
              'baseball glove','skateboard','surfboard','tennis racket','bottle','plate','wine glass','cup','fork','knife',
              'spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
              'couch','potted plant','bed','mirror','dining table','window','desk','toilet','door','tv','laptop','mouse','remote'
              ,'keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','blender','book','clock','vase','scissors',
              'teddy bear','hair drier','toothbrush','hair brush']

model = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' #Take model and graph to network
cikarim_grafigi = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(cikarim_grafigi, model) #Load model and graph to network

net.setInputSize(320, 320) #Set Input Size
net.setInputScale(1.0 / 127.5) #Multiplier for frame values.
net.setInputMean((127.5, 127.5, 127.5)) #Scalar with mean values which are subtracted from channels.
net.setInputSwapRB(True) #BGR TO RGB

while True:
    success, img = cap.read() #Read the screen
    '''
    net.detect(img, confThreshold=thres)
    returns 3 values
    
    classIds: The Id of the detected object. Function returns an Id
    confs: Confidence of the detected Id.
    bbox: The 2 Coordinates of the detected object. We'll use this later the putText on Object.
    '''
    classIds, confs, bbox = net.detect(img, confThreshold=thres) #Detect object

    '''
    print classIds and bbox to see everything
    '''
    print(classIds, bbox)
    '''
    if len(classIds) = 0:
    code may give some errors. So we need to check that. 
    '''
    if len(classIds) != 0:
        '''
        zipping ang flattening values to easily get these values.
        '''
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            '''
            create rectangles to the given coordinates.
            '''
            cv2.rectangle(img, box, color=(255, 255, 0), thickness=2)
            try:
                '''
                Putting texts to the given coordinates
                '''
                cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
            except:
                print('Unable to do')
    '''
    this is output screen and delay 1 sec.
    '''
    cv2.imshow('Output', img)
    cv2.waitKey(1)
