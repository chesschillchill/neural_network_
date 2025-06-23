#include "neural_network.h"
#include "read_mnist.h"

#define TRAIN_IMAGE_PATH "fashion/train-images-idx3-ubyte"
#define TRAIN_LABEL_PATH "fashion/train-labels-idx1-ubyte"
#define TEST_IMAGE_PATH "fashion/t10k-images-idx3-ubyte"
#define TEST_LABEL_PATH "fashion/t10k-labels-idx1-ubyte"
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

    vector<float> result;
    NeuralNetwork nn(train_images[0].size(), 128, 64, 10);

    result = nn.forward(train_images[0]);

    // For demonstration, we will just print the first result
    cout << "First forward result: ";
    for (const auto &value : result)
    {
        cout << value << " ";
    }
    cout << endl;
    return 0;
}