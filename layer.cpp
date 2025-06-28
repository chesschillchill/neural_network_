#include "layer.h"

void Layer::forward(const vector<float>& inputs) {
	for (auto& node : nodes) {
		node.forward(inputs);
		node.setA(ActivationFunction::sigmoid(node.getZ()));
	}
}

vector<float> Layer::backward(const vector<float>& dA, const vector<float>& pre_layer, float learning_rate) {
	vector<float> dZ;
	dZ.resize(nodes.size(), 0.0f);

	for (size_t i = 0; i < nodes.size(); ++i) {
		dZ[i] = dA[i] * ActivationFunction::sigmoidDerivative(nodes[i].getZ());
	}

	for (size_t i = 0; i < nodes.size(); ++i) {
		vector<float> dWeights;
		for (size_t j = 0; j < pre_layer.size(); ++j) {
			dWeights.push_back(dZ[i] * pre_layer[j]);
		}
		nodes[i].updateParameters(dWeights, dZ[i], learning_rate);
	}

	vector<float> dA_prev(pre_layer.size(), 0.0f);
	for (size_t i = 0; i < nodes.size(); ++i) {
		for (size_t j = 0; j < pre_layer.size(); ++j) {
			dA_prev[j] += dZ[i] * nodes[i].getWeight()[j];
		}
	}

	return dA_prev;
}

// OUTPUT LAYER
void OutputLayer::forward(const vector<float>& inputs) {
	vector<float> output_values;
	for (auto& node : nodes) {
		node.forward(inputs);
		output_values.push_back(node.getZ());
	}

	output_values = ActivationFunction::softmax(output_values);
	
	for (size_t i = 0; i < nodes.size(); ++i) {
		nodes[i].setA(output_values[i]);
	}
}

vector<float> OutputLayer::backward(const vector<float>& dA, const vector<float>& pre_layer, float learning_rate) {
	vector<float> dZ;
	dZ.resize(nodes.size(), 0.0f);
	
	vector<float> output = getAllZ();
	output = ActivationFunction::softmaxDerivative(output);
	for (size_t i = 0; i < nodes.size(); ++i) {
		dZ[i] = dA[i] * output[i];
	}

	for (size_t i = 0; i < nodes.size(); ++i) {
		vector<float> dWeights;
		for (size_t j = 0; j < pre_layer.size(); ++j) {
			dWeights.push_back(dZ[i] * pre_layer[j]);
		}
		nodes[i].updateParameters(dWeights, dZ[i], learning_rate);
	}
	
	vector<float> dA_prev(pre_layer.size(), 0.0f);
	for (size_t i = 0; i < nodes.size(); ++i) {
		for (size_t j = 0; j < pre_layer.size(); ++j) {
			dA_prev[j] += dZ[i] * nodes[i].getWeight()[j];
		}
	}

	return dA_prev;
}