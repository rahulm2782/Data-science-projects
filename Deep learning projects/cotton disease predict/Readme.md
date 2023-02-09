## <font color='red'>Cotton Disease Predictor
A Deep Learning model built using Tensorflow's Keras library to classify images of diseased and fresh cotton leaves and plants.

<b><u>Requirements</b></u>
- Tensorflow 2.x
- Keras
- OpenCV
- Numpy
- Vgg16
    
The model is built using the VGG16 architecture as the base model and adding custom layers on top of it. The VGG16 model is pre-trained on the imagenet dataset and its final fully connected layer is removed. The output of the VGG16 model is flattened and fed into a dense layer with 512 units and ReLU activation. The final layer is a dense layer with 4 units and softmax activation, which is used for classification.

Data Augmentation
The training data is augmented using the ImageDataGenerator class from the Tensorflow's Keras library. The data augmentation techniques used include rescaling, rotation, horizontal flip, shear and zoom.

Training and Validation
The model is trained on the training data and validated on the validation data. The loss function used is categorical crossentropy and the optimizer is Adam. The model is trained for 10 epochs.

Evaluation
The final model achieved a loss of 0.168 and an accuracy of 93.6% on the validation data.