# Autism-Sprectrum-detection-using-Face-recognization

## abstract

Observation plays a key role in identifying
the mental disorders, especially in infants and
children of the age zero to eight years. Autism
Spectrum Disorder (ASD) tends to develop
majorly during this period of time and parents
and clinicians are expected to be keenly observant of the movements of the child. Kaggle
dataset was taken and transfer learning was applied on the same. We also used VGGNet to
train the model and get better accuracy.
Keywords: Autism Spectrum Disorder, VGGNet, Transfer Learning, clinicians

##  Proposed Methodology


Initially the required libraries have been imported.
The weights of the base model that has been
applied in ResNet taken from Imagenet. With a
rotation range of 90, the Image Data Generator
has been used. Two dense layers with relu
activation function has been used, followed by
a flatten layer and dropout layer of 0.5. Fully
connected layers are of size 1024 each. The model
is compiled with learning rate 0.00001 and losse
function categorical cross entropy. The number of
epochs were 50 and batch size is given to be 8.
The graphs have been hence plotted.
An xception model for fine tuning where the
base model is of the shape (299,299,3) was done.
The base layers were then freezed for no further
changes. By freezing it means that the layer will
not be trained. Hence the weights will not be
changed.There is no enough time to train the deep
neural networks.
The base layer is freezed followed by a flatten
layer, this is followed by a dense layer. The activation function here is relu and this is followed by
another dense layer followed by a prediction layer
which is the output layer with softmax activation
function.
Then a xception model is prepared to carry out
finetuning. When freeze baselayers is True, the
base model acts as a feature extractor that is
used for classification by the latter layers. Model
summary has hence been derived.
Categorical crossentropy with varying epochs and
optimizers has been tried out. The model was then
compiled and Relevent graphs have been hence
drawn

#  Experimentation


* In Xception model, we carried out experiments by
changing the number of epochs and optimizers.
The following are the summaries of each model.
EPOCHS=1, OPTIMIZER=Adam
The epoch value is taken as 1 and
the optimizer is taken as Adam. The
learning rate here taken is 0.001.

* EPOCHS = 2,OPTIMIZER = Adam
The epoch value is taken as 2 and the optimizer
is taken as Adam. The learning rate here taken is
0.001

* EPOCHS = 3,OPTIMIZER = RMSprop
The epoch value is taken as 3 and the optimizer is
taken as RMSProp. The learning rate here taken
is 0.001

* EPOCHS = 3,OPTIMIZER = SGD
The epoch value is taken as 3 and the optimizer
is taken as SGD. The learning rate here taken is
0.001


# result

With ResNet model we got training accuracy
of 0.5802 and with xception model the highest
accuracy we got through various hyperparameter
tuning was 0.62701.
Graphs have been provided for reference
Optimizer=Adam, Epoch=3
Optimizer=RMSprop, Epoch=3

# Conclusion

So as we conclude the models which we have tried
are discussed above and it has to be noted that
xception model gave a good accuracy and with
these we were able to get a good amount of results.
But still this leaves a huge gap and opportunity to
work on.








