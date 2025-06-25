#pragma once
#include <algorithm>
#include <memory>

#include "layer.h"
#include "activation_function.h"

using namespace std;

class NeuralNetwork
{
private:
    vector<shared_ptr<Layer>> layers; // Contain hidden layers and output layer
public:
    NeuralNetwork()
    {
    }

    NeuralNetwork(unsigned char image_size, vector<unsigned char> list_hidden_layers_node, unsigned char label_size)
    {
       list_hidden_layers_node.insert(list_hidden_layers_node.begin(), image_size); // Add input layer size at the beginning
       list_hidden_layers_node.push_back(label_size); // Add output layer size at the end
	   // Example for list_hidden_layers_node: {784, 128, 128, 10}

	   // Start from index 1, the first hidden layer, which connects to the input layer
       // End 
       for (int i = 1; i < list_hidden_layers_node.size() - 1; ++i)
       {
           Layer tempLayer = Layer(list_hidden_layers_node[i], list_hidden_layers_node[i - 1]);
		   layers.push_back(make_unique<Layer>(tempLayer));
       }

       size_t size = list_hidden_layers_node.size();
	   OutputLayer outputLayer = OutputLayer(list_hidden_layers_node[size - 1], list_hidden_layers_node[size - 2]);

	   layers.push_back(make_unique<OutputLayer>(outputLayer));
    }

    ~NeuralNetwork() = default;

    vector<shared_ptr<Layer>> getLayer()
	{
        return layers;
	}

    //vector<float> forward(const vector<float> &train_image);

    /*
	Make a true label vector for the neural network.
	For example, if the label is 3, the vector will be:
	[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    */
    vector<unsigned char> createTrueLabelVector(unsigned char label);

    void backward(const vector<float> &predict_label, const vector<unsigned char> &true_label);

    void training(const vector<vector<float>>& train_images,
        const vector<unsigned char>& train_labels,
        int num_epoch,
        float learning_rate);
};
