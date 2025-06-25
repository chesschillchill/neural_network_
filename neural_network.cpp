#include "neural_network.h"

vector<float> NeuralNetwork::forward(const vector<float> &train_image)
{
    vector<float> hidden1NodesValue = hiddenLayer1.calculateValues(train_image);
    hidden1NodesValue = ActivationFunction::relu(hidden1NodesValue);

    vector<float> hidden2NodesValue = hiddenLayer2.calculateValues(hidden1NodesValue);
    hidden2NodesValue = ActivationFunction::relu(hidden2NodesValue);

    vector<float> outputNodesValue = outputLayer.calculateValues(hidden2NodesValue);
    outputNodesValue = ActivationFunction::softmax(outputNodesValue);

    return outputNodesValue;
}

vector<float> NeuralNetwork::createTrueLabelVector(unsigned char label)
{
	vector<float> true_label(labelSize, 0.0f);
	true_label[label] = 1.0f; // Set the index corresponding to the label to 1
	return true_label;
}

float NeuralNetwork::calculateCost(const vector<float>& predict_label, unsigned char label) {
	float cost = 0.0f;

	for (size_t i = 0; i < predict_label.size(); ++i) {
		if (i == label) {
			cost += pow((1 - predict_label[i]), 2); // True label
		}
		else {
			cost += pow(predict_label[i], 2); 
		}
	}
	return cost;
}

void NeuralNetwork::backward(const vector<float> &predict_label, const vector<float> &true_label)
{

}