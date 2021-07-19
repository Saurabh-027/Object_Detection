import cv2

# img = cv2.imread('D:\python\Object_Detection\cartoon.jpg')
capt = cv2.VideoCapture(0)
capt.set(3, 640)
capt.set(4, 480)

className = []
classFile = 'D:\python\Object_Detection\coco.names'
with open(classFile, 'rt') as f:
    className = f.read().rstrip('\n').split('\n')

configPath = 'D:\python\Object_Detection\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'D:\python\Object_Detection\\frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = capt.read()
    classIds, confid, bobox = net.detect(img, confThreshold=0.6)
    print(classIds, bobox)


    if len(classIds) != 0:
          for classId, confidence, box in zip(classIds.flatten(), confid.flatten(), bobox):
                cv2.rectangle(img, box, color=(0, 255, 255), thickness=3)
                cv2.putText(img, className[classId - 1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 0, 255), 2)

    cv2.imshow("Output", img)
    cv2.waitKey(1)
