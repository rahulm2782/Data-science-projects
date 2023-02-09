# Toxic Comment Classification Model
This code is a deep learning model built using the Tensorflow framework to classify toxic comments into 6 categories: toxic, severe_toxic, obscene, threat, insult, identity_hate.

### Data
The model is trained on the 'hate_speech.csv' dataset which contains comments and the corresponding label for each of the 6 categories.

### Preprocessing
The comments are preprocessed by converting them into numerical vectors using TextVectorization from the Tensorflow library. This is done in the following steps:

###Split the data into training, validation, and testing datasets
Set the maximum number of words in the vocabulary to 200000
Vectorize the text using TextVectorization
Preprocessthe data using map, cache, shuffle, batch, and prefetch

### Model
The model is built using the Keras API of Tensorflow and has the following architecture:

An Embedding layer with a maximum number of words in the vocabulary and an output of 32 dimensions
A Bidirectional LSTM layer with 32 units and a tanh activation function
Three fully connected layers with 128, 256, and 128 units respectively and a ReLU activation function
A final layer with 6 units and a sigmoid activation function
Evaluation
The model is trained for 1 epoch with the binary cross-entropy loss and Adam optimizer. The model's accuracy is evaluated using precision, recall, and accuracy metrics. The results are plotted using the Matplotlib library.

### Deployment
The model is saved and loaded to make predictions on new comments. The predictions are made by vectorizing the input comment and passing it through the model. The results are returned as a string with the predicted probabilities of each of the 6 categories.

The code also includes an implementation of a web interface using the GradIO library that takes in a comment and returns the predicted probabilities of each category.




