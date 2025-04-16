### Context
Our client had an interesting proposal put forward to them, and requested our help to assess whether it was viable.

At a recent tech conference, they spoke to a contact from a robotics company that creates robotic solutions that help other businesses scale and optimise their operations.

Their representative mentioned that they had built a prototype for a robotic sorting arm that could be used to pick up and move products off a platform. It would use a camera to "see" the product, and could be programmed to move that particular product into a designated bin, for further processing.

The only thing they hadn't figured out was how to actually identify each product using the camera, so that the robotic arm could move it to the right place.

We were asked to put forward a proof of concept on this - and were given some sample images of fruits from their processing platform.

If this was successful and put into place on a larger scale, the client would be able to enhance their sorting & delivery processes.

![cnn-fruit-classification-title-img](https://github.com/user-attachments/assets/a2948887-bc13-4e1a-95f2-438b9c570de2)


### Actions
We utilise the Keras Deep Learning library for this task.

We start by creating our pipeline for feeding training & validation images in batches, from our local directory, into the network. We investigate & quantify predictive performance epoch by epoch on the validation set, and then also on a held-back test set.

Our baseline network is simple, but gives us a starting point to refine from. This network contains 2 Convolutional Layers, each with 32 filters and subsequent Max Pooling Layers. We have a single Dense (Fully Connected) layer following flattening with 32 neurons followed by our output layer. We apply the relu activation function on all layers, and use the adam optimizer.

Our first refinement is to add Dropout to tackle the issue of overfitting which is prevalent in the baseline network performance. We use a dropout rate of 0.5.

We then add in Image Augmentation to our data pipeline to increase the variation of input images for the network to learn from, resulting in a more robust results as well as also address overfitting.

With these additions in place, we utlise keras-tuner to optimise our network architecture & tune the hyperparameters. The best network from this testing contains 3 Convolutional Layers, each followed by Max Pooling Layers. The first Convolutional Layer has 96 filters, the second & third have 64 filters. The output of this third layer is flattened and passed to a single Dense (Fully Connected) layer with 160 neurons. The Dense Layer has Dropout applied with a dropout rate of 0.5. The output from this is passed to the output layer. Again, we apply the relu activation function on all layers, and use the adam optimizer.

Finally, we utilise Transfer Learning to compare our network's results against that of the pre-trained VGG16 network.



### Results
We have made some huge strides in terms of making our network's predictions more accurate, and more reliable on new data.

Our baseline network suffered badly from overfitting - the addition of both Dropout & Image Augmentation elimited this almost entirely.

In terms of Classification Accuracy on the Test Set, we saw:

Baseline Network: 75%
Baseline + Dropout: 85%
Baseline + Image Augmentation: 93%
Optimised Architecture + Dropout + Image Augmentation: 95%
Transfer Learning Using VGG16: 98%
Tuning the networks architecture with Keras-Tuner gave us a great boost, but was also very time intensive - however if this time investment results in improved accuracy then it is time well spent.

The use of Transfer Learning with the VGG16 architecture was also a great success, in only 10 epochs we were able to beat the performance of our smaller, custom networks which were training over 50 epochs. From a business point of view we also need to consider the overheads of (a) storing the much larger VGG16 network file, and (b) any increased latency on inference.



### Growth/Next Steps
The proof of concept was successful, we have shown that we can get very accurate predictions albeit on a small number of classes. We need to showcase this to the client, discuss what it is that makes the network more robust, and then look to test our best networks on a larger array of classes.

Transfer Learning has been a big success, and was the best performing network in terms of classification accuracy on the Test Set - however we still only trained for a small number of epochs so we can push this even further. It would be worthwhile testing other available pre-trained networks such as ResNet, Inception, and the DenseNet networks.



