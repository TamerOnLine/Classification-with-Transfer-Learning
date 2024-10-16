
# Image Classification Using Transfer Learning

This project demonstrates image classification using a pre-trained model (VGG16) through transfer learning. The CIFAR-10 dataset is used, which consists of 60,000 32x32 color images in 10 different classes.

## Overview

In this project, we leverage the power of transfer learning by using the VGG16 model, pre-trained on the ImageNet dataset, to classify images from the CIFAR-10 dataset. The final layers of the model are fine-tuned to fit our classification task.

## Dataset

The CIFAR-10 dataset is used in this project. It includes the following:
- 50,000 training images
- 10,000 test images
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Model Architecture

- **Base Model**: VGG16 pre-trained on ImageNet
- **Fine-tuned Layers**: The last layers of VGG16 are replaced with fully connected layers tailored for the CIFAR-10 classification task.
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy

## Data Preprocessing

- Resizing images to 32x32 pixels to fit the model's input requirements.
- One-hot encoding of the labels.
- Data augmentation using rotation, zoom, shift, and flip techniques to prevent overfitting.

## Training

- **Batch size**: 64
- **Epochs**: 25
- **Callbacks**: Early stopping and model checkpointing are used to save the best model and avoid overfitting.
- **Data Augmentation**: Applied to enhance the model's ability to generalize.

## Results

The model achieved the following performance metrics:
- **Training Accuracy**: `XX%`
- **Validation Accuracy**: `XX%`
- **Training Loss**: `XX`
- **Validation Loss**: `XX`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/YourRepoName.git
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the model training, use the following command:
```bash
python train_model.py
```

You can find the saved model and training logs in the `models/` directory.

## Conclusion

This project demonstrates how transfer learning can be effectively used for image classification tasks. By leveraging pre-trained models, we achieve high accuracy with less computational power and time.

## License

This project is licensed under the MIT License.

## Acknowledgments

- The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
