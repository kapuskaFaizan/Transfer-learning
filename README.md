# Computer-Vision TRANSFER LEARNING FOR FACIAL IMAGE RECOGNITION MODE
Transfer learning is the idea of overcoming the isolated learning paradigm and utilizing knowledge acquired for one task to solve related ones. For this project we have extracted features for new custom facial images belonging to 5 multiple classes using pre-trained VGG-model. And have used these features to train a brand new deep neural network thus utilizing the power of transfer learning for our model efficiency. 
# Accuracy 
We have trained this model to recognize my face as well as faces of 4 other people. The accuracy attained by the model is 75% which is quite fine but could be increased if larger data set would be used.

# loss function
The loss function used in this model is sparse_categorical_crossentropy instead of catagorical_crossentropy since the latter expects hot encoded labels. Both of these loss functions are good for multiclass single labeled targets but since we have used integer targets sparse_categorical_crossentropy seemed to be an optimal choice.
