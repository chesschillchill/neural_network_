#include "neural_network.h"

vector<float> NeuralNetwork::forward(vector<float> train_image)
{
    vector<float> hidden1NodesValue = hiddenLayer1.calculateValues(train_image);
    hidden1NodesValue = ActivationFunction::relu(hidden1NodesValue);

    vector<float> hidden2NodesValue = hiddenLayer2.calculateValues(hidden1NodesValue);
    hidden2NodesValue = ActivationFunction::relu(hidden2NodesValue);

    vector<float> outputNodesValue = outputLayer.calculateValues(hidden2NodesValue);
    outputNodesValue = ActivationFunction::softmax(outputNodesValue);
    return outputNodesValue;
}