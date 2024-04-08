from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import numpy as np
import cv2

COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True) 
model.eval()



def get_prediction(img_path, threshold):
  """
  get_prediction
    parameters:
      - img_path - path of the input image
      - threshold - threshold value for prediction score
    method:
      - Image is obtained from the image path
      - the image is converted to image tensor using PyTorch's Transforms
      - image is passed through the model to get the predictions
      - class, box coordinates are obtained, but only prediction score > threshold
        are chosen.
     
  """
  img = Image.open(img_path)
  transform = T.Compose([T.ToTensor()])
  img = transform(img)
  pred = model([img])
  pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
  pred_score = list(pred[0]['scores'].detach().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  return pred_boxes, pred_class



def object_detection_api(img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3): 
  boxes, pred_cls = get_prediction(img_path, threshold) 
#   print("boxes 1 : ",boxes)
  modified_boxes = []
  for val in boxes:
    l = list()
    for ele in val:
      l.append((int(ele[0]),int(ele[1])))
    modified_boxes.append(l)
  boxes = modified_boxes
#   print("boxes 2 : ",boxes)
  # Get predictions 
  img = cv2.imread(img_path) 
  # Read image with cv2 
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
  # Convert to RGB 
  for i in range(len(boxes)): 
    # print("boxes : ",boxes)
    cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) 
    # Draw Rectangle with the coordinates 
    cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) 
    # Write the prediction class 
    # plt.figure(figsize=(20,30)) 
    # # display the output image 
    # plt.imshow(img) 
    # plt.xticks([]) 
    # plt.yticks([])
    # plt.show()
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
  name = img_path.split(".")
  name[0] = name[0]+"_detection"
  save_path = '.'.join(name)
  cv2.imwrite(save_path,img)

# object_detection_api("people.jpg", threshold=0.8)
# object_detection_api('./car.jpg', rect_th=6, text_th=5, text_size=5)
object_detection_api('traffic.jpg', rect_th=2, text_th=1, text_size=1)
