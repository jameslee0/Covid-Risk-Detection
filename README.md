Covid Risk Detection is an open source repository that provides object tracking, social distance detection, and mask detection.

## Results
<img src="images/outpy.gif" width="750">

Above the model is tracking to see if everyone is social distancing, wearing a face mask, and tracking each person.<br>

For the social distancing, the bounding boxes around the person will turn red for those who aren't and stay green for those who are<br>

For the Face Mask Detection, the bounding boxes are located around each persons face. A white box indicates that it is unknown if the person is wearing a face mask,
the green box indicates the person is wearing a face mask, and the red box indicates the person is not wearing a face mask.

Each person has an unique id that tracks their information throughout the frame. The tracking algorithm can find issues with occlusion, but will quickly correct itself afterwards. Each unique id is specified by the green dot in the center of each person.
## Installation
To use the covid guidelines detection software is simple. All you need to do is clone this repository and download the required modules:
```
git clone https://github.com/jameslee0/guidelines_detection.git
```
Once the repository is cloned, navigate to the main directory and install the required modules:
```
pip install -r requirements.txt
pip install --upgrade tensorflow
```
The installation of tensorflow and the requirements text might be different based on your os. The following references for different operating systems are listed below:
- [tensorflow installation](https://www.tensorflow.org/install/pip)
- [Requirements.txt](https://pip.pypa.io/en/stable/reference/pip_install/#requirements-file-format)

## Use
Once all the required modules are installed properly, the model files need to be downloaded. YOLOV3 was used for the object detection and a custom model was used 
for the face mask detection. All the models need to be downloaded and placed in the ```models``` directory. The following models must be downloaded and added to the directory:
- [YOLOV3](https://pjreddie.com/media/files/yolov3.weights) **Clicking the link will download the weights**
- [Face Mask Detection](https://www.dropbox.com/sh/31i8o4cc2p89oqw/AADO4pm50JzxGdwGUNCKngVBa?dl=0)
  * Unzip and add entire folder into ```models``` directory
  
After downloading and adding the files to their location, the ```models``` directory should look like so:
```
.
└── models
    ├── mask_detection
    ├── coco.names
    ├── yolov3.cfg
    └── yolov3.weights
```
To use the software you'll need video's to test the software on. If the user has no video available, they can download
a traditional test video for object detection:
```
pip install youtube_dl
youtube-dl https://www.youtube.com/watch?v=pk96gqasGBQ -o street.mp4
```
This will download [this](https://www.youtube.com/watch?v=pk96gqasGBQ) youtube video that works for testing. <br>

In order to use the software, a video must be specified. Using that recently downloaded video, the software can be quickly accessed:
```
python covid_tracker.py -v street.mp4
```
Assuming everything else is correctly set up, this command should run the entire software correctly.

#### Software features
To save the video use the ```[-s save]``` flag
```
python covid_tracker.py -v street.mp4 -s True
```

To only use the social distancing tool use the ```[-sd socialdistance]``` flag
```
python covid_tracker.py -v street.mp4 -sd True
```

To only use the face mask detection tool use the ```[-fm facemask]``` flag
```
python covid_tracker.py -v street.mp4 -fm True
```

## Face Mask Detection Model
The Face Mask Detection model was trained on the following [model](https://github.com/jameslee0/guidelines_detection/blob/main/Training_Model/face_mask_detection.ipynb). This model was trained using transfer learning from the [BiT](https://blog.tensorflow.org/2020/05/bigtransfer-bit-state-of-art-transfer-learning-computer-vision.html). Any user with an existing dataset can train their own Face Mask Detection model with the script provided.
