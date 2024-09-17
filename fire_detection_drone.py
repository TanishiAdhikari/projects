!pip install roboflow --quiet #installing roboflow library
import os
HOME = os.getcwd()  #get the os current working directory
print(HOME)

!pip install ultralytics==8.0.20

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

from ultralytics import YOLO
from IPython.display import display, Image

!mkdir {HOME}/datasets
%cd {HOME}/datasets
from roboflow import Roboflow
rf = Roboflow(api_key="hmycfBEaws2Scmch9bDe")
project = rf.workspace("fire-detecting-drone").project("fire-detection-bmuaj")
dataset = project.version(1).download("yolov8")

%cd {HOME}
!yolo task=detect mode=train model=yolov8m.pt data={dataset.location}/data.yaml epochs=20 imgsz=800 plots=True  #run yolo

!ls {HOME}/runs/detect/train/

%cd {HOME}
Image(filename=f'{HOME}/runs/detect/train/confusion_matrix.png', width=600)     #display the confusion matrix

%cd {HOME}
Image(filename=f'{HOME}/runs/detect/train/results.png', width=600)      #displaying training graphs

%cd {HOME}
Image(filename=f'{HOME}/runs/detect/train/val_batch0_pred.jpg', width=600)         #validation batch 0

%cd {HOME}
!yolo task=detect mode=val model={HOME}/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml     #run yolo for object detection validation

%cd {HOME}
#running yolo for object detection prediction
!yolo task=detect mode=predict model={HOME}/runs/detect/train/weights/best.pt conf=0.25 source={dataset.location}/test/images save=True       #conf-confidence threshold

import glob
from IPython.display import Image, display

for image_path in glob.glob(f'{HOME}/runs/detect/predict/*.jpg')[:3]:     #iterating through first 3 predicted image paths
      display(Image(filename=image_path, width=600))
      print("\n")

import pickle
pickle.dump(HOME, open('model.pkl', 'wb'))

!pip install roboflow

!pip install ultralytics==8.0.196


from roboflow import Roboflow
rf = Roboflow(api_key="hmycfBEaws2Scmch9bDe")

project = rf.workspace("fire-detecting-drone").project("fire-detection-bmuaj")

version = project.version(1)

version.export("yolov8")
# version.download("yolov8", path=r"C:\Users\student\Downloads")

import os

# Specify the directory path
download_dir = r"C:\Users\student\Desktop"

# Create the directory if it doesn't exist
os.makedirs(download_dir, exist_ok=True)

# Change the current working directory to the desired destination
os.chdir(download_dir)

# Download the dataset version
version.download("yolov8")

import shutil
import os

# Download the dataset to default location
version.download("yolov8")

# Specify the source and destination paths
source_path = os.path.join(os.getcwd(), "Fire-Detection-1")
destination_path = r"C:\Users\student\Downloads"  # Adjust this path

# Move the dataset to the desired directory
shutil.move(source_path, destination_path)


