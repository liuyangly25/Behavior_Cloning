# **Behavioral Cloning** 

## Writeup Report

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_img/NVIDIA_model.png "Model Visualization"
[image2]: ./report_img/center_driving.jpg "Center Line Driving"
[image3]: ./report_img/recover1.jpg "Recovery Image"
[image4]: ./report_img/recover2.jpg "Recovery Image"
[image5]: ./report_img/recover3.jpg "Recovery Image"
[image6]: ./report_img/flip1.jpg "Normal Image"
[image7]: ./report_img/flip2.jpg "Flipped Image"
[image8]: ./report_img/bridge.jpg "Bridge Image"
[image9]: ./report_img/dirt.jpg "Dirt Image"
[image10]: ./report_img/model_loss.png "Loss Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 48, and 3x3 filter sizes and depths of 64 (model.py lines 69-73) 

The model includes RELU layers to introduce nonlinearity (code line 69 - 73), and the data is normalized in the model using a Keras lambda layer (code line 66). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 78). Also the model collected a large amount of data to reduce overfitting (22,626 images). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 21). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 81).

The model also used 3 cameras, which has correction for left and right cameras (model.py line 35)

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, bridge driving, dirts avoiding, sharp turn driving, and smooth driving.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the VGG. I thought this model might be appropriate because it has a good accuracy in ImageNet.

However, it yielded a big .h5 file. It also had poor control for the car.

Then, I choosed the CNN similar to [NVIDIA approach](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). It has a good result in the real road test, which being said in the paper.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding a dropout layer right after the last convolutional layer. And I tuned the keep_prob rate to be 0.25. 

Then I added an output full connected layer of size 1, because our output for this project is the turning angle, which is just one number. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, such as bridge, sharp turns, track-dirt-road intersections. To improve the driving behavior in these cases, I collected additional datas only at these locations.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 68-84) consisted of a convolution neural network with the following layers and layer sizes 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   						| 
| Normalization     	| Lambda Function 								|
| Convolution 5x5     	| 2x2 stride 									|
| RELU					| Activation function							|
| Convolution 5x5     	| 2x2 stride 									|
| RELU					| Activation function							|
| Convolution 5x5     	| 2x2 stride 									|
| RELU					| Activation function							|
| Convolution 3x3     	|  												|
| RELU					| Activation function							|
| Convolution 3x3     	| 			 									|
| RELU					| Activation function							|
| Dropout				| Keep probability 0.25 						|
| Flatten 				| Flatten Layer									|
| Fully connected		| output 100									|
| Fully connected		| output 50										|
| Fully connected		| output 10										|
| Output 				| output 1										|

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![NVIDIA Model][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center Driving][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to go back to good driving behavior. These images show what a recovery looks like starting from road margin back to center:

![Recover Step 1][image3]
![Recover Step 2][image4]
![Recover Step 3][image5]

To augment the data sat, I also flipped images and angles thinking that this would creat more training data and solving the "too much left turn" issue for track 1. For example, here is an image that has then been flipped:

![Normal Image][image6]
![Flipped Image][image7]

Bridges seems different from the normal track. So I capture additional bridge images for training. For example, here is an image from bridge image sets.

![Bridge Image][image8]

And also for a section of the track without yellow line. See an image below.

![Dirt Image][image9]

After the collection process, I had 7,542 number of data points. I then preprocessed this data by using normalization, adding left and right cameras and data augmentation. So, the total number of images is 45,252.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by proventing overfitting. See the loss figure below. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![Loss Image][image10]

The result is very good but not perfect. Here is my suggestion for my future work. 

1. Capture additional data on the part which car did not stay in center of lanes. This is to make car always staying in the center of two lanes.

2. Try to decrease input image size to increase training speed without lossing accuracy.

