# Traffic Sign Classification Model

This project is a machine learning-based classification model to identify traffic signs. It uses a convolutional neural network (CNN) to classify traffic signs based on images provided by the user. The model is trained using labeled traffic sign images and can predict the class of any new image of a traffic sign.

##Features

- Train a CNN model for traffic sign classification.
- Evaluate the model's accuracy on validation data.
- Save the best model during training.
- Make predictions for new traffic sign images.
- Display predictions on images with labels.

##Project Setup

## 1. Clone the Repository

git clone <repository_url>
cd traffic_sign_classification

## 2. Install Dependencies

You can install the required dependencies using pip:

--pip install -r requirements.txt

requirements.txt includes:
- tensorflow for model building and training
- numpy for data manipulation
- pandas for handling datasets
- matplotlib for visualizations
- scikit-learn for label encoding and splitting data
- PIL for image processing

## 3. Dataset

The dataset consists of traffic sign images stored in folders. Each folder corresponds to a specific class of traffic sign, and the images in each folder are labeled with the class name.

The structure should look like this:

myData/
    |-- 0/
        |-- image1.jpg
        |-- image2.jpg
    |-- 1/
        |-- image1.jpg
        |-- image2.jpg
    |-- ...

##4. Preprocessing Data

In the preprocessing step, images are resized to 64x64 and normalized by dividing the pixel values by 255.0. Labels are encoded using LabelEncoder to convert class names to integers.

##5. Model Architecture

The model is a Convolutional Neural Network (CNN) built using Keras. The architecture includes:
- Convolutional layers followed by max-pooling layers
- Flattening the output to feed into fully connected layers
- Output layer with softmax activation to predict the class

## 6. Training the Model

The model is trained using the preprocessed images and corresponding labels. The best model is saved based on validation loss using the ModelCheckpoint callback.

## 7. Testing and Prediction

The model can be tested with new images of traffic signs. Simply provide the path to the image, and the model will predict the class of the traffic sign.

-- test_image_path = 'path_to_your_test_image.jpg'
-- test_image = load_img(test_image_path, target_size=(64, 64))
-- test_image = img_to_array(test_image) / 255.0
-- test_image = np.expand_dims(test_image, axis=0)

-- predicted_class = model.predict(test_image)
-- predicted_label = label_encoder.inverse_transform([np.argmax(predicted_class)])

-- print(f"Predicted class: {predicted_label[0]}")

## 8. Saving the Model

The trained model can be saved using Keras' save() method. The best model will automatically be saved during training.

-- model.save('traffic_sign_model.h5')

## 9. Visualizing the Results

You can visualize the predicted class by overlaying the prediction label on the test image.

-- draw = ImageDraw.Draw(test_image_pil)
-- label = f"Predicted: {predicted_label[0]}"
-- draw.text((10, 10), label, fill=(255, 0, 0))

-- plt.imshow(test_image_np)
-- plt.axis('off')
-- plt.show()

License

This project is licensed under the MIT License
