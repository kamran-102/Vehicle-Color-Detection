# Vehicle Color Detection Model

### Overview

This project focuses on training a convolutional neural network (CNN) to detect and predict the color of vehicles from images. The model leverages data augmentation techniques to enhance generalization and is designed to classify images into one of 15 predefined color categories.

## Dataset Organization
The dataset is structured into directories for each vehicle color. Separate folders are used for training and validation data.

- **Training Directory:** `train`
- **Validation Directory:** `val`

## Model Architecture
The model consists of the following layers:
- **Convolutional Layers:** Three layers using `Conv2D` with ReLU activation for feature extraction.
- **Pooling Layers:** Three `MaxPooling2D` layers to reduce spatial dimensions.
- **Fully Connected Layers:** A `Flatten` layer followed by a dense layer with 512 units and a `Dropout` of 0.5.
- **Output Layer:** A dense layer with 15 units and `softmax` activation for multiclass classification.

## Training Process
The model is compiled using the Adam optimizer and categorical cross-entropy loss. Data augmentation is applied to the training set to improve model robustness.

- **Batch Size:** 32
- **Epochs:** 25
- **Augmentation Techniques:** Rotation, width/height shift, shear, zoom, horizontal flip.

## Model Evaluation and Save
The trained model achieves an accuracy of approximately `80.39%` on the validation set. The performance is measured using the `evaluate` method on the validation data and saving it for future use like (`vehicle_color_model_detection.h5`).

## How to Use the Model
-   Use `Testing_VCM.ipynb` file to predict the color of target image using the saved model. 

### Dependencies

- **Python 3.x**
- **TensorFlow**
- **Keras**
- **Numpy**
- **Matplotlib**
- **PIL**


### Installation

1. **Install Python 3.x:** Ensure that Python 3.x is installed on your system.
2. **Install Required Libraries:** Run the following command to install the necessary Python libraries:
    ```bash
    pip install tensorflow keras pillow numpy matplotlib
    ```
3. **Clone the Repository:** Download or clone this repository to your local machine.


### Notes

- Ensure that your dataset is properly organized and cleaned before starting the training process. Incorrectly labeled or corrupted images can negatively impact model performance.
- It's recommended to run the training on a machine with a GPU for faster computation.
- Hyperparameters like the number of layers and batch size can be fine-tuned for better accuracy depending on the dataset size and complexity.


### Disclaimer

This script is provided for educational purposes only, and the author assumes no liability for any misuse.
