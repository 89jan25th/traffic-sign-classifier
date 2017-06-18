# **Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./pictures/pic1.JPG "summary"
[image2]: ./pictures/pic2.JPG "images"
[image3]: ./pictures/pic3.JPG "histogram"
[image4]: ./new_picture/29.jpg "Bycicle crossing"
[image5]: ./new_picture/30.jpg "Beware of ice/snow"
[image6]: ./new_picture/27.jpg "Pedestrians"
[image7]: ./new_picture/28.jpg "Children crossing"
[image8]: ./new_picture/23.jpg "Slippery road"
[image9]: ./pictures/pic4.JPG "softmax table"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Hello, here is my [code](Traffic-Sign_classifier.ipynb)



### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

My code is at the second tab. I used DataFrame function from the pandas library to make a table of summary.
![alt text][image1]


#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

My code is at third tab.

First, I used matplotlib.pyplot's basic functions and random's random function to show training set examples randomly.
It was important for me to see how the images from the set look like generally.
![alt text][image2]

Then I used 'hist' from matplotlib.pyplot to see the histogram of data sets. It is interesting that the histogram is not so fair but biased. However the training set and testing set has a similar distribution.
![alt text][image3]



### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

My code is at the fourth tab.

I tried histogram equalization and normalization from OpenCV library but it seemed that there was some problems with data type with that, so I just did RGB to gray conversion and normalization with numpy.

The reasone why I chose these schemes is that I found many of the pictures over or under exposed and not focused very well. I didn't do blur because the pictures are already blurred enough. 



#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)


The set was given as 34,799 for training, 12,630 for testing, and 4,410 for validating. 
The training set is 7.8 x validating set, or validating set/training set = 12%. I did research about which ratio is the best but I couldn't find the one best rule except that 7:3=training:validating is quite popular. However, at the same time, lowering the validating set from 4,410 can be too low for the set, so I decided to get back here after finishing the network.
As a conclusion, I didn't get back to this part because I met 0.94. But I believe tweaking the sets can yield the better result.
Also I didn't generate additional data. I had several ideas like rotating or tilting to take advantages of assymetry of some symbols. However, I did a little bit of research and found that manipulation doesn't really produce a better result.



#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

At "Model Architecture" tab, there is my code.

I built my network based on LeNet-5 example from the course.

My final model consisted of the following layers:  
  
| Layer               		|     Description	        		                 			| 
|:---------------------:|:---------------------------------------------:| 
| Input               		| 32x32x1 gray image   			                  				| 
| Convolution 1x1      	| 1x1 stride, valid padding, outputs 32x32x32   |
| RELU			             		|						                                   						|
| Convolution 5x5		  			|	1x1 stride, valid padding, outputs 28x28x32   |
| Max pooling	         	| 2x2 stride,  outputs 14x14x32 			            	|
| Convolution 5x5  	    | 1x1 stride, valid padding, outputs 10x10x64   |
| RELU					             |					                                   							|
| Max pooling	         	| 2x2 stride,  outputs 5x5x64               				|
| Dropout			          		|		keep_prob = 0.5				                    						|
| Fully connected		     | input 1600  output 400	                      	|
| RELU		             			|						                                   						|
| Dropout				          	|		keep_prob = 0.5		                    								|
| Fully connected	     	| input 400  output 84		                        |
| RELU					             |							                                   					|
| Fully connected		     | input 84  output 43                         		|				
| Softmax               | 	                                             |			 
 


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

This part is also included in the Model Architecture tab.
For epoch, I tried to figure out when it converges by seeing the accuracy at each epoch. I found 50 is reasonable and enough for my network.
For batch number, I changed it in a range of 50 to 250, and just settled with 128, which is from LeNet-5 example. I found it tends to be better in the neighborhood of 100, so I just used 128.
For optimizer, I did research and AdamOptimizer is the most popular one recently.
For learning rate, I used one from LeNet-5 as well.

Shortly, I used most of variables from LeNet-5 example. It just worked fine without changing it so much. However I believe there is room for improvement here as well.



#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My code is at from 'Model Architecture' to 'Train, Validate and Test the Model' tab.

My final model results were:
* training set accuracy of 0.993
* validation set accuracy of 0.952
* test set accuracy of 0.937

I built my neural network based on LeNet-5 example from the course.

I added three dropout layers to reduce overfitting. I could see the accuracy gap between the training set and validating set. After adding dropout layers, the gap reduced also later I could find less outlying predictions for test set from interet also reduced. 

I put 1x1 convnet at the very first filter. I was stuck at 0.88 ~ 0.90 accuracy, so I did many trials and errors based on research I did. Then I found this method can easily raise my networks' accuracy to 0.937. 



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

| Image			           |     Image	        		                  			| 
|:---------------------:|:---------------------------------------------:| 
| ![alt text][image4]   		| ![alt text][image5]                   									| 
| ![alt text][image6]    		| ![alt text][image7]                 									| 
|  ![alt text][image8] 		|                   									| 



I tried to find images with different backgrounds(sky, grass, ...) and from diffrent angles. Any kind of noises(bad quality, over or under exposed, out-focused) will affect the network.

1) bycicle: this one is easy one.
2) ice: tilted, rotated, grass in the background
3) pedestrian: bent, darkned, the whity-blue colors are similar in the sign and sky
4) chlidren: complex image in the background, cropped at the end
5) slippery road; out-focused, and the sign is relatively small


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code is at 'Predict the Sign Type for Each Image' tab.

My network yields different results at every session.run and I couldn't figure it out yet.
The accuracy is most frequently 0.8, but sometimes 1.0 and rarely 0.6 and 0.4.

Even though my network's prediction model on new internet images is not stable, I will continue discussing for the sake of the argument.
At the instance below, my network mistook Pedestrians sign with General caution, and from this I can assume that my network recognizes triangle sign with one vertically long black line in the center, but it finally failed to distingush them.

Here are the results of the prediction:  
| Image			              |     Prediction	        		                  			| 
|:---------------------:|:---------------------------------------------:| 
| Bicycles crossing   		| Bicycles crossing                    									| 
| Beware of ice/snow 			| Beware of ice/snow 				                 						|
| Pedestrians	       			| General Caution			                    								|
| Children crossing	   	| Children crossing					 		                   		|
| Slippery road		      	| Slippery Road                          							|



#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

My code is at 'Output Top 5 Softmax Probabilities For Each Image Found on the Web.

![alt text][image9]

For first image, there is no actual value in top 5. 
For second and fifth image, the prediction value is almost 1.0000 to the actual value.
For third and fourth image, the actual value is in the top 5 but the probability is still quite low as 0.000xx. 

From this result, I can conclude that my network mostly assigne the most points to the top 1 whether it's the actual value or not. Considering that my network scored 0.93 with the given test data set, I assume that there is some noise in the process between loading an internet image to input it to my network.
