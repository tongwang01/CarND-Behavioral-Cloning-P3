# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py: code to load data, build and train model
* drive.py: code for driving the car in autonomous mode
    * The only changed I've made to the orginal drive.py script was changing the `set_speed` parameter from `9` to `30`, mostly to speed up iteration speed.
* model.h5: final model 
    * In addition, in the folder `nvidia_12_working/` there are the checkpoints of this model at each epoch during training.
* video.mp4: video recording of the car driving track 1 for two laps, using `model.h5`
* writeup_report.md: write up of this project. You are reading this.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

To run `model.py`, use the following command line arguments:
```sh
python model.py \
--model=nvidia \
--model_dir=12 \
--epochs=10 \
--data_dirs="../beta_simulator_mac/data/center_driving_1/driving_log.csv" \
--data_dirs="../beta_simulator_mac/data/center_driving_2/driving_log.csv" \
--data_dirs="../beta_simulator_mac/data/center_driving_reverse_1/driving_log.csv" \
--data_dirs="../beta_simulator_mac/data/curves_1/driving_log.csv"
```

`model`: one of "vgg" or "nvidia", the two models implemented.

`model_dir`: specific a name for the directory to store the trained model.

`epochs`: number of epochs to train the model.

`data_dirs`: a list of `driving_log.csv` files that we want to train the model on; can pass this argument multiple times to train on multiple files.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My (final, best performaning) model is based on this [NVIDIA convolutional neutral network model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/).  

Here's a summary of the model architecture:

```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636     
_________________________________________________________________
dropout_1 (Dropout)          (None, 20, 77, 36)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928     
_________________________________________________________________
dropout_2 (Dropout)          (None, 4, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 8448)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               1081472   
_________________________________________________________________
dense_2 (Dense)              (None, 64)                8256      
_________________________________________________________________
dense_3 (Dense)              (None, 32)                2080      
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 33        
=================================================================
Total params: 1,223,189
Trainable params: 1,223,189
Non-trainable params: 0
```

Additionally, I've also implemented and trained a model with VGG architecture. However, despite being a much larger model (about 10x the number of parameters of the NVIDIA model), this model performs significantly worse. It also takes much longer to train. My hypotheses are that 1) VGG is prone to overfitting when applied to this problem as the input image space is relatively homogenous, and 2) the extra pooling and dropout layers added too much noise and made it difficult for gradients to backpropogate through the network.  

#### 2. Attempts to reduce overfitting in the model

There are three main techniques applied to prevent overfitting:
* Two Dropout layers
* Early stopping in training
* Attempts are made to collect a comprehensive and diverse set of training examples (more on this below).


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

Other parameters of the model were tuned manually, by looking at the driving performance of resulting models.


#### 4. Appropriate training data

I generated training data from the following simulator runs on track 1:
* Three laps of my "best effort" driving
* Two laps of my "best effort" driving, in reverse direction
* Additional runs for a couple of sharp that the model had trouble navigating initally

I also used the following techniques to augment training data:
* Flipping images and steering angles
* Using images from left and right cameras
    * This turned out to be particularly effective in teaching the model how to recover when its trajectory deviates from the center of the lane.
    * I set the correction angle to be 0.15, based on some manual tuning.



### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I first developed the data input / model output modules, using a very simple model as a placeholder, in order to ensure that the pipeline is functional from end to end. This model performed miserably but that is as expected.

I then beefed up my model by going to the VGG architecture. As discussed above, this appears to have overshot the target as the model is very slow to train, and more importantly the trained model performs badly on the track. The car would frequently go off track, fall into water, etc.

Therefore I decided to use a simpler mode, i.e. the NVIDIA model introduced in the lectures. This model has done wonders. Within a few epochs of training the car was able to navigate large portions of the track. 

I then augmented the training data by leveraging images from left and right cameras - again as discussed above, this proved to be a critical step in teaching the car how to recover when it begins to go off track. After this the car was above to drive around track 1 smoothly.

I added a few dropout layers to the original NVIDIA model to prevent overfitting. This did not have a noticable impact on driving performance on track 1, but (ostensibly) may make the model more generalizable to other scenarios.


#### 2. Final Model Architecture

See the "1. An appropriate model architecture has been employed" section above for a detailed description of the model architecture.

#### 3. Creation of the Training Set & Training Process

For training data creation see section " 4. Appropriate training data" above.

I applied early stopping in training, and set the `restore_best_weights` parameter to `True` so that Keras automatically updates the final model weight to that from the best performing epoch.

One interesting thing I observed is that below a certain threshold, validation loss does not correlate strongly with driving performance on the tracks. More specifically, the model from some epoch could a higher validation loss than one from another epoch, nonetheless its actual driving behavior on the track appears to be smoother, and more natural to how a human would drive the vehicle. My hypothesis is that, unlike e.g. image classification, there are multiple ways to control a car so that it does the right thing on the tracks. Turning right one fraction of a second sooner or later probably does not matter all that much. Therefore in this case the loss function functions more as a proxy metric, rather than the true task-success objective function 

