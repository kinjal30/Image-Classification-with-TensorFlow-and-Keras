# Image-Classification-with-TensorFlow-and-Keras

# AML- Image Classification with TensorFlow and Keras

This repository contains code for an image classification which focuses on building a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images into three classes: Cars, Footwear, and Trees.

## Code Structure:

- **AML_Image Classification.ipynb:** Jupyter Notebook containing the entire code.
  
- **/content/drive/MyDrive/AML:** Directory containing the dataset and necessary files.

    - **/train:** Directory for training images.
    - **/val:** Directory for validation images.
    - **/test:** Directory for test images.

## Instructions:

1. **Dataset Setup:**
    - The dataset is structured into training, validation, and test sets.
    - Images are categorized into three classes: Cars, Footwear, and Trees.

2. **Data Preprocessing:**
    - Data augmentation and preprocessing are performed using TensorFlow's ImageDataGenerator.

3. **Model Architecture:**
    - Transfer learning is applied using the ResNet50 pre-trained model.
    - The top layers of ResNet50 are replaced with custom layers for fine-tuning.
    - The model is compiled with the Adam optimizer and categorical crossentropy loss.

4. **Training:**
    - The model is trained on the training set with early stopping on the validation set to prevent overfitting.

5. **Evaluation:**
    - Model performance is evaluated on the test set, and accuracy metrics are reported.

6. **Results:**
    - Correctly and incorrectly classified images are visualized for further analysis.

## Dependencies:

- Python 3.x
- TensorFlow 2.x
- Matplotlib

## Usage:

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/aml-image classification.git
    ```

2. Open the Jupyter Notebook `AML_Image Classification.ipynb` in a compatible environment.

3. Run the cells sequentially to train the model and visualize results.
