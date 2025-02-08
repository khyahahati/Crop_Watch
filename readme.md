# Crop Watch 🌱

Crop Watch is a plant disease detection system that utilizes a Convolutional Neural Network (CNN) model for early detection and better crop management. This project aims to help farmers and agricultural professionals identify plant diseases quickly and accurately to improve crop yields and reduce losses.

## Architecture

So, here's how the Crop Watch system works behind the scenes:

1. **Image Processing**: When you upload a plant image, we first resize it to 128x128 pixels. This makes it easier for our model to analyze the image. We also convert it into a format that our Convolutional Neural Network (CNN) can understand.

2. **Convolutional Neural Network (CNN)**: At the heart of our system is a CNN, which is like the brain of the operation! It has several layers:
   - **Conv2D and MaxPooling2D Layers**: These layers work together to spot patterns and features in the images. Think of them as filters that help highlight important details.
   - **Dropout Layers**: We’ve added these to prevent overfitting, which means our model won’t just memorize the training data but will actually learn to recognize diseases in new images too.

3. **Training Phase**: 
   - We use the Adam optimizer for efficient training, tweaking the model’s weights to minimize the errors between what it predicts and the actual results.
   - The training happens over 10 epochs with a batch size of 32. Basically, we feed the model a group of 32 images at a time to help it learn better.
   - At the end of training, we use a softmax layer to turn the model’s raw outputs into easy-to-understand probabilities for each disease class.

4. **Performance Analysis**: Once the model is trained, we put it to the test! We evaluate its performance using a separate set of images to see how well it can identify diseases it hasn’t seen before.


This architecture works really well and has been trained on the [Plant Disease Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset?select=New+Plant+Diseases+Dataset(Augmented)). 


## Installation
   ```bash
   pip install -r requirements.txt

   #for using the prediction model only
   streamlit run app.py

   
   #for using the prediction and chatbot app
   streamlit run chat.py
