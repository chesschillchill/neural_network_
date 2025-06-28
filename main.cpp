#include "neural_network.h"
#include "read_mnist.h"

#define TRAIN_IMAGE_PATH "fashion/train-images-idx3-ubyte"
#define TRAIN_LABEL_PATH "fashion/train-labels-idx1-ubyte"
#define TEST_IMAGE_PATH "fashion/t10k-images-idx3-ubyte"
#define TEST_LABEL_PATH "fashion/t10k-labels-idx1-ubyte"

/*
Neural Network Configuration:
- Input Layer: 784 nodes (28x28 pixels)
- Hidden Layer 1: 128 nodes
- Hidden Layer 2: 64 nodes
- Output Layer: 10 nodes (for 10 classes)
*/
#define IMAGE_SIZE 784 // 28x28 pixels
#define LABEL_SIZE 10  // 10 classes for Fashion MNIST
#define HIDDEN_LAYER1_SIZE 128
#define HIDDEN_LAYER2_SIZE 64

#define TRAIN_SIZE 60000 // Number of training images
#define TEST_SIZE 10000  // Number of test images

#define LEARNING_RATE 0.1f
#define NUM_EPOCH 5
int main()
{
    string train_images_path = TRAIN_IMAGE_PATH;
    string train_labels_path = TRAIN_LABEL_PATH;
    string test_images_path = TEST_IMAGE_PATH;
    string test_labels_path = TEST_LABEL_PATH;

    vector<vector<float>> train_images;
    vector<unsigned char> train_labels;
    vector<vector<float>> test_images;
    vector<unsigned char> test_labels;

    ReadDataset::read_mnist_images(train_images_path, train_images);
    ReadDataset::read_mnist_labels(train_labels_path, train_labels);
    ReadDataset::read_mnist_images(test_images_path, test_images);
    ReadDataset::read_mnist_labels(test_labels_path, test_labels);

	NeuralNetwork nn(IMAGE_SIZE, { HIDDEN_LAYER1_SIZE, HIDDEN_LAYER2_SIZE }, LABEL_SIZE);
	nn.training(train_images, train_labels, test_images, test_labels, NUM_EPOCH, LEARNING_RATE);

    return 0;
}