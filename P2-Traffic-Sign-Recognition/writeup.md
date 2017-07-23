# Traffic Sign Recognition

The project code can be found in this [link](https://github.com/camigord/Self-Driving-Car-Nanodegree/blob/master/P2-Traffic-Sign-Recognition/Traffic_Sign_Classifier.ipynb). The TensorFlow model can be downloaded from [here](https://github.com/camigord/Self-Driving-Car-Nanodegree/tree/master/P2-Traffic-Sign-Recognition/model2).

[//]: # (Image References)

[image1]: ./assets/train_samples.jpg "Training samples"
[image2]: ./assets/samples_per_categorie.jpg "Number of examples per categorie"
[image3]: ./assets/test_samples.jpg "Testing examples"

## Build a traffic sign recognition project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Summarizing the dataset

This is a short summary describing the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). The code for extracting this information from the dataset is located in the second code cell of the Ipython [notebook](https://github.com/camigord/Self-Driving-Car-Nanodegree/blob/master/P2-Traffic-Sign-Recognition/Traffic_Sign_Classifier.ipynb).

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32x32x3)
* The number of unique classes/labels in the data set is 43

### Visualizing and exploring the dataset.

Lets start by visualizing some of the training samples and their corresponding labels:

![alt text][image1]

We can also analyse how are the different labels distributed across the training, validation and testing datasets. The following image shows how many examples of each one of the 43 different categories are present on each dataset. It is possible to see that although these distributions are not uniform, the proportion of samples on each dataset is very similar.

![alt text][image2]


### Data preprocessing.

In my first try, I decided to only normalize the data without any further preprocessing. My idea was that the color channels may provide additional information to the particular task of classifying traffic road signs. This step was performed directly on TensorFlow employing the function _tf.image.per_image_standardization()_ on every image in the current batch as shown below:

```
  x = tf.placeholder(tf.float32, (None, 32, 32, 3))
  x_norm = tf.map_fn(lambda img: tf.image.per_image_standardization(img), x)
```

The problem with this approach was that it did not generalize very well to the test images taken from the internet, getting a very low accuracy (~20%). I believe that because of the color information, the network may overfit too easily to the training data. Effects like illumination and the general background of the image may have a strong effect in the classification accuracy. 

In my second try and after tuning the model, I also tried to convert the images to grayscale. Again, the preprocessing was applied using TensorFlow and the function _tf.image.rgb_to_grayscale()_ which is capable of operating directly on image batches.

```
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
# To Grayscale
x_gray = tf.image.rgb_to_grayscale(x)
# Normalization
x_norm = tf.map_fn(lambda img: tf.image.per_image_standardization(img), x_gray)
```
<!--
Here is an example of a traffic sign image before and after grayscaling/normalization.
--->

### Model Architecture

The final model is based on the [VGG architecture](https://arxiv.org/pdf/1409.1556.pdf). It consists of 4 convolutional layers arranged as shown below. For simplicity, I have removed the activation function from the table, but all the layers (convolutions and fully connected layers) are followed by a _ReLU_ activation.

<table>
  <tr>
    <td align="center"><b>Layer</b></td>
    <td align="center"><b>Description</b></td>
  </tr>
  <tr>
    <td align="center">Input</td>
    <td align="center">32x32x1 Grayscale image</td>
  </tr>
  <tr>
    <td align="center">Convolution 3x3</td>
    <td align="center">1x1 stride, valid padding, outputs 30x30x16</td>
  </tr>
  <tr>
    <td align="center">Convolution 3x3</td>
    <td align="center">1x1 stride, valid padding, outputs 28x28x16</td>
  </tr>
  <tr>
    <td align="center">Max pooling</td>
    <td align="center">2x2 stride,  valid padding, outputs 14x14x16</td>
  </tr>
  <tr>
    <td align="center">Convolution 3x3</td>
    <td align="center">1x1 stride, valid padding, outputs 12x12x24</td>
  </tr>
  <tr>
    <td align="center">Convolution 3x3</td>
    <td align="center">1x1 stride, valid padding, outputs 10x10x24</td>
  </tr>
  <tr>
    <td align="center">Max pooling</td>
    <td align="center">2x2 stride,  valid padding, outputs 5x5x24</td>
  </tr>
  <tr>
    <td align="center">Fully connected</td>
    <td align="center">size = 400</td>
  </tr>
  <tr>
    <td align="center" colspan="2">Dropout</td>
  </tr>
  <tr>
    <td align="center">Fully connected</td>
    <td align="center">size = 120</td>
  </tr>
  <tr>
    <td align="center" colspan="2">Dropout</td>
  </tr>
  <tr>
    <td align="center">Fully connected</td>
    <td align="center">size = 43</td>
  </tr>
  <tr>
    <td align="center" colspan="2">Softmax</td>
  </tr>
</table>

### Training the model

The model was trained using the following parameteres: 

| Parameter			    |     Value	        		| 
|:-----------------:|:---------------------:| 
| Optimizer      		| Adam   								| 
| Learning rate     | 0.0001   							|
| Batch size      	| 128   							  |	 
| Epochs      	    | 30   									| 
| Dropout      	    | 0.8   								|

The figure below shows the learning curve of the model. The number of epochs was tuned so that training stops when the performance of the model does not improve significantly any further. The learning rate was initially set to 0.01, but it was tuned by observing the learning curve. The batch size was set to 128 because it is large enough to provide an informative gradient while still matching the computational power I had at hand. Dropout was set to 0.5, but the model was not learning fast enough; increasing the value to 0.8 improved the training time and validation accuracy. 

<img src="./assets/training_curve.jpg" width="750" height="350" />

#### Tuning the model 

My final model results were:

| Dataset			      |     Accuracy	        | 
|:-----------------:|:---------------------:| 
| Training set      | 99.7%  								| 
| Validation set    | 97.2%   							|
| Testing set      	| 94.5%  							  |	 

As mentioned above, I implemented the model based on the [VGG architecture](https://arxiv.org/pdf/1409.1556.pdf), which was a state of the art classification model a couple of years ago. The main idea behind this architecture is that we can replace a large receptive field convolutional layer with a stack of very small convolutional filters. The authors of the original paper demonstrated that this not only reduces the number of parameters in the model, but also improves the performance by making the classification model more discriminative. 

 - Initially I trained the model without dropout, which resulted in the model quickly overfitting to the training set when using a learning rate of 0.01. 
 - In my second attempt I introduced dropout with a probability of 50% after every fully connected layer and reduced the learning rate to 0.0001. This model, however, did not learn as fast as expected and was converging towards a training accuracy of around 60% after 40 epochs.
 - In my third attempt, and based on the previous results, I modified the dropout layers by keeping 80% of the activations while still using a small learning rate (0.0001). My logic was that the model may had not been complex enough to generalize to the data when using only 50% of the activations. This last model converged to the accuracies presented above in less than 30 epochs.
 
The reason why I decided to use the VGG architecture is because it is very similar to LeNet and very easy to implement on TensorFlow. Moreover, I was confident that this architecture should easily outperform LeNet provided proper tuning. If LeNet can achieve a classification accuracy of around 89% on this dataset, I was sure that a version of the VGG architecture improved by recent techniques like Dropout should improve these values and easily reach an accuracy of 93%.
 
### Testing the model on new images

I collected 10 different traffic signs from the web, some of which are shown below with their corresponding labels.

![alt text][image3]

Images X and Y may be harder to classify given that they are partially occluded by snow or leaves respectively. 

#### Performance on this 'new' testing set
2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:-----------------:|:---------------------------------:| 
| Wild animal      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly classify 8 out of 10 traffic signs, which gives an accuracy of 80%. This may seem like a low accuracy compared to the 94.5% on the testing set, but we need to consider that we did not train the model with partially occluded samples. It is actually surprising that the model is capable of correctly classify one of the occluded examples (__image X__) given that half of the sign is covered by snow. If we ignore the missclassification of __image Y__ where the stop sign is mostly hidden from view, the model would achieve an accuracy of 90%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


