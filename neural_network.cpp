#include "neural_network.h"

//vector<float> NeuralNetwork::forward(const vector<float> &train_image)
//{
//    vector<float> hidden1nodesvalue = hiddenlayer1.calculatevalues(train_image);
//    hidden1nodesvalue = ActivationFunction::relu(hidden1nodesvalue);
//
//    vector<float> hidden2nodesvalue = hiddenlayer2.calculatevalues(hidden1nodesvalue);
//    hidden2nodesvalue = ActivationFunction::relu(hidden2nodesvalue);
//
//    vector<float> outputnodesvalue = outputlayer.calculatevalues(hidden2nodesvalue);
//    outputnodesvalue = ActivationFunction::softmax(outputnodesvalue);
//
//    return outputnodesvalue;
//}

vector<unsigned char> NeuralNetwork::createTrueLabelVector(unsigned char label)
{
	size_t labelSize = layers.back()->getNodes().size(); // Get the size of the output layer
	vector<unsigned char> true_label(labelSize, 0);
	true_label[label] = 1; // Set the index corresponding to the label to 1
	return true_label;
}

void NeuralNetwork::backward(const vector<float> &predict_label, const vector<unsigned char> &true_label)
{

}

void NeuralNetwork::training(const vector<vector<float>>& train_images,
	const vector<unsigned char>& train_labels,
	int num_epoch,
	float learning_rate)
{
	for (size_t i = 0; i <= num_epoch; ++i) {
		cout << "Epoch " << i + 1 << endl;
		for (auto train_image : train_images) {
			
		}
	}
}