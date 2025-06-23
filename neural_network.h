#pragma once
#include <iostream>
#include <vector>

#include "layer.h"
#include "activation_function.h"

using namespace std;

class NeuralNetwork
{
private:
    int imageSize;
    int labelSize;

    Layer hiddenLayer1;
    Layer hiddenLayer2;

    Layer outputLayer; // Assuming an output layer is needed, though not used in the constructor
public:
    NeuralNetwork()
    {
        imageSize = 0;
        labelSize = 0;
        hiddenLayer1 = Layer();
        hiddenLayer2 = Layer();
        outputLayer = Layer();
    }

    NeuralNetwork(int image_size, int num_hidden_nodes1, int num_hidden_nodes2, int label_size)
    {
        imageSize = image_size;
        labelSize = label_size;

        hiddenLayer1 = Layer(num_hidden_nodes1, image_size);
        hiddenLayer2 = Layer(num_hidden_nodes2, num_hidden_nodes1);
        outputLayer = Layer(label_size, num_hidden_nodes2);
    }

    ~NeuralNetwork() = default;

    vector<float> forward(const vector<float> &train_image);

    void backward(const vector<float> &predict_label, const vector<float> &true_label);
};
