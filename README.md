# Simpson-classifier

Simpson character classifier with CNN model using Keras

**Dataset**

The original dataset is : </br>
https://www.kaggle.com/alexattia/the-simpsons-characters-dataset </br>
Kaggle's dataset include 20 classes but these are not of equal size. Since, I dropped from 20 to 16 classes in order to balance it.

For model selection, the dataset has been split in train/validation/test set : 
* **[Train set]**      12688 images includes in *dataset/train*;
* **[Validation set]** 3619 images includes in *dataset/validation*;
* **[Test set]**       1829 images includes in *dataset/test*;

# Net output examples
Examples of net output after model training using new images :

| Abraham             |  Apu | Bart | Chief Wiggum |
:-------------------------:|:-------------------------: | :-------------------------: | :-------------------------: |
![](https://github.com/Dantekk/Simpson-classifier/blob/main/predictions_examples/Abraham.jpg)  |  ![](https://github.com/Dantekk/Simpson-classifier/blob/main/predictions_examples/Apu.jpg) |  ![](https://github.com/Dantekk/Simpson-classifier/blob/main/predictions_examples/Bart.jpg) |  ![](https://github.com/Dantekk/Simpson-classifier/blob/main/predictions_examples/Chief_Wiggum.jpg)

| Comic Book Guy             |  Edna Krabappel | Homer | Krusty the clown |
:-------------------------:|:-------------------------: | :-------------------------: | :-------------------------: |
![](https://github.com/Dantekk/Simpson-classifier/blob/main/predictions_examples/Comic%20Book%20Guy.jpg)  |  ![](https://github.com/Dantekk/Simpson-classifier/blob/main/predictions_examples/Edna%20Krabappel.jpg) |  ![](https://github.com/Dantekk/Simpson-classifier/blob/main/predictions_examples/Homer.jpg) |  ![](https://github.com/Dantekk/Simpson-classifier/blob/main/predictions_examples/Krusty%20the%20clown.jpg)

| Lisa             |  Marge | Milhouse | Krusty the clown |
:-------------------------:|:-------------------------: | :-------------------------: | :-------------------------: |
![](https://github.com/Dantekk/Simpson-classifier/blob/main/predictions_examples/Lisa.jpg)  |  ![](https://github.com/Dantekk/Simpson-classifier/blob/main/predictions_examples/Marge.jpg) |  ![](https://github.com/Dantekk/Simpson-classifier/blob/main/predictions_examples/Milhouse.jpg) |  ![](https://github.com/Dantekk/Simpson-classifier/blob/main/predictions_examples/Moe%20Szyslak.jpg)

| Montgomery Burns             |  Ned Flanders | Principal Skinner | Sideshow Bob |
:-------------------------:|:-------------------------: | :-------------------------: | :-------------------------: |
![](https://github.com/Dantekk/Simpson-classifier/blob/main/predictions_examples/Montgomery%20Burns.jpg)  |  ![](https://github.com/Dantekk/Simpson-classifier/blob/main/predictions_examples/Ned%20Flanders.jpg) |  ![](https://github.com/Dantekk/Simpson-classifier/blob/main/predictions_examples/Principal%20Skinner.jpg) |  ![](https://github.com/Dantekk/Simpson-classifier/blob/main/predictions_examples/Sideshow%20Bob.jpg)

# Model Architecture
After model selection, this model architecture has obtained best result :

* Input Data Shape: 200x200x3

Layer 1 :
* Convolutional Layer : 64 filter with kernel size (3x3)
* Activation Function: ReLu
* Max Pooling : Pool shape (2x2) with stride (2x2)
* Dropout Rate: 0.4

Layer 2 :
* Convolutional Layer : 128 filter with kernel size (3x3)
* Activation Function: ReLu

Layer 3 :
* Convolutional Layer : 128 filter with kernel size (3x3)
* Activation Function: ReLu
* Max Pooling : Pool shape (2x2) with stride (2x2)
* Dropout Rate: 0.4

Layer 4 :
* Convolutional Layer : 256 filter with kernel size (4x4)
* Activation Function: ReLu
* Max Pooling : Pool shape (2x2) with stride (2x2)
* Dropout Rate: 0.4

Classification:
* Flatten
* Dense size 512 -> Activation Function: ReLu
* Dropout Rate: 0.4
* Dense size 256 -> Activation Function: ReLu
* Dropout Rate: 0.4
* Dense size 128 -> Activation Function: ReLu
* Dropout Rate: 0.4
* Dense size 16 -> Activation Function: Softmax

**Optimizer** : Adam

**Loss** : Categorical Crossentropy

**NOTE**

I have applied data augmentation on the training set using ImageDataGenerator Keras method.

# Model performance

After model selection, these are model performance on train and test set : 

Set image type | Loss value | Accuracy value |
-------------- | ---------- | -------------- |
Train| 0.1521 | 0.9581 |
Test | 0.2402 | 0.9470 |

It's possible increase the performance spending time on model selection and training model on more epochs (I trained model on 100 epochs).

# Workflow
- Use *model_training.py* for training model and test it.
- Use *model_predict.py* for to make predictions from new images.

# Important Notes:
* Used Python Version: 3.6.0
* Used Tensorflow Version: 2.3.1
* Used OpenCV Version: 4.1.2
