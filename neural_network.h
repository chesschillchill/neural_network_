#pragma once
#include <iostream>
#include <vector>
#include <math.h>

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

	// Use this Layer to calculate the output of the neural network
	// by function Layer::calculateValues(const vector<float> inputs)
	Layer outputLayer; 
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

    /*
	Make a true label vector for the neural network.
	For example, if the label is 3, the vector will be:
	[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    Set value as float for calculation.
    */
    vector<float> createTrueLabelVector(unsigned char label);

	float calculateCost(const vector<float>& predict_label, unsigned char label);

    void backward(const vector<float> &predict_label, const vector<float> &true_label);
};
