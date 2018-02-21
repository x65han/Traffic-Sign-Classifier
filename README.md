# Traffic Sign Classification

<img  align="right" src="https://camo.githubusercontent.com/ee91ac3c9f5ad840ebf70b54284498fe0e6ddb92/68747470733a2f2f7777772e74656e736f72666c6f772e6f72672f696d616765732f74665f6c6f676f5f7472616e73702e706e67" width="100px" />
<img src="https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg"/>

- 96% accuracy on *German Traffic Sign Classification*
- `Tensorflow` + `OpenCV` + [LeNet 5 model](http://yann.lecun.com/exdb/lenet/)

Overview
---

In this project, I used deep neural networks and [Lenet 5](http://yann.lecun.com/exdb/lenet/) convolutional neural networks to classify traffic signs. I trained and validated a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I then try out your model on images of German traffic signs that I found on the web.

<div align="center"><a href="http://yann.lecun.com/exdb/lenet/"><b>Lenet 5 Convolutional Neural Network</b></a></div>
<div align="center"><img src="https://github.com/x65han/Traffic-Sign-Classifier/blob/master/miscellaneous/lenet.png?raw=true" width="80%" /></div>

Build & Run
---
- run `./run.sh`. This script downloads the data sets and triggers the python program
- or run `jupyter notebook Traffic_Sign_Classifier.ipynb`. This notebook has more details and steps.

## Data Set Summary & Exploration
* The size of training set is `31367`
* The size of the validation set is `7842`
* The size of test set is `12630`
* The shape of a traffic sign image is `(32, 32, 3)`
* The number of unique classes/labels in the data set is `43`

## Data Visualization

### Label Histogram
<div align="center"><img src="https://github.com/x65han/Traffic-Sign-Classifier/blob/master/miscellaneous/bar_graph.png?raw=true" width="80%" /></div>

### Sample Data with sign code
<div align="center"><img src="https://github.com/x65han/Traffic-Sign-Classifier/blob/master/miscellaneous/sample_input.png?raw=true" height="60%" width="100%" /></div>

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The first step is to shuffle the data. The reason for shuffling is quite obvious and to promote generalization over memorization. A normalization step is also added to improve the accuracy by about 10%. The last step but not the least is to convert images to grayscale. Converting the image to grayscale reduces the input image size from 32x32x3 to 32x32x1. The colors in a traffic sign is not too important. Hence feeding grayscale data would not hurt the accuracy, but improves processing speed and reduces computer hardware requirement.

**Here is a sample collection of images after `pre-processing`**

<div align="center"><img src="https://github.com/x65han/Traffic-Sign-Classifier/blob/master/miscellaneous/pre_process_data.png?raw=true" height="60%" width="100%" /></div>


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   				    |
| Convolution 5x5    	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x16   |
| Fully connected		| Output 43    									|
| Softmax				| one_hot_y, logits => cross_entropy			|
| Cost Function			| reduce_mean of cross_entropy					|
| Optimizer				| AdamOptimizer									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I played around with batch_size, learning_rate, EPOCHS and adding dropout layers. Adding dropout layers made the learning process too slow for my computer to handle, hence removed. After playing around with the other hyperparameters, I decided that 128 batch_size, 0.001 learning_rate and 20 EPOCHS is the best for my model.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ? (Not sure where to fin)
* validation set accuracy of 96%
* test set accuracy of 88.9% 

I basically used the original Lenet architecture. I noticed that my training set accuracy is high and validation set accuracy is low. I realized that my CNN has more memorization on the training model, rather than generalization. Hence my CNN is not doing as well with the validation set. I tried to add a dropout layer into my CNN. The learning time turns out to be too much for my computer to handle, hence given up.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are some German traffic signs that I found on the web:

<div align="center"><img src="https://github.com/x65han/Traffic-Sign-Classifier/blob/master/miscellaneous/my_test_images.png?raw=true" height="60%" width="100%" /></div>

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

I have achieved 100% accuracy on my test images.

| Code               	|     Name       	        					| 
|:---------------------:|:---------------------------------------------:| 
| 25        			| Road Work      								| 
| 38     				| Keep right     								|
| 34					| Turn left ahead								|
| 1	      		     	| Speed limit (30km/h)			 				|
| 12				    | Priority road     							|
| 11				    | Right-of-way at the next intersection      	|			
| 3				        | Speed limit (60km/h) 							|
| 18				    | General caution    							|

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.

| Probability           |     Name       	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100%        			| Road Work      								| 
| 92.10%  				| Keep right     								|
| 88.02%				| Turn left ahead								|
| 97.69%   		     	| Speed limit (30km/h)			 				|
| 95.47%			    | Priority road     							|
| 99.72%			    | Right-of-way at the next intersection      	|
| 98.76%		        | Speed limit (60km/h) 							|
| 37.73%			    | General caution    							|

All signs have 90%+ confidence rating, except the `General caution` sign. My guess is that the `General caution` sign is too pixelated to make an accurate prediction.

<div align="center"><img src="https://github.com/x65han/Traffic-Sign-Classifier/blob/master/miscellaneous/prob.png?raw=true" width="80%" /></div>
