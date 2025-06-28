#include "neural_network.h"

using namespace std;

void NeuralNetwork::forward(const vector<float> &train_image)
{
	vector<float> inputs = train_image; // Start with the input layer

	for (auto& layer : layers) {
		layer->forward(inputs); // Forward pass through each layer

		inputs.clear();
		inputs = layer->getAllA(); 
	}
}

vector<unsigned char> NeuralNetwork::createTrueLabelVector(unsigned char label)
{
	size_t labelSize = layers.back()->getNodes().size(); // Get the size of the output layer
	vector<unsigned char> true_label(labelSize, 0);
	true_label[label] = 1; // Set the index corresponding to the label to 1
	return true_label;
}

void NeuralNetwork::backward(const vector<float>& train_image, const vector<float> &predict_label, const vector<unsigned char> &true_label, float learning_rate)
{
	vector<float> dA;
	for (size_t i = 0; i < predict_label.size(); ++i) {
		dA.emplace_back(predict_label[i] - float(true_label[i]));
	}

	for (size_t i = layers.size() - 1; i > 0; --i) {
		vector<float> pre_layer; 
		if (i == 0) 
			pre_layer = train_image; // For the first layer, use the input image
		else 
			pre_layer = layers[i - 1]->getAllA();
		dA = layers[i]->backward(dA, pre_layer, learning_rate);
	}

	// debug
    vector<float> outputWeights = layers.front()->getNodes()[0].getWeight(); 
	for (size_t i = 0; i < 10; ++i) {
		cout << outputWeights[i] << " "; // Print the weights of the first node in the output layer
	}
	cout << endl;
	cout << endl;
}

bool NeuralNetwork::getAccuracy(const vector<float>& predict_label, unsigned char true_label){
	int max_index = 0;
	for (size_t i = 1; i < predict_label.size(); ++i) {
		if (predict_label[i] > predict_label[max_index]) {
			max_index = i; // Find the index of the maximum value
		}
	}
	
	if (max_index == true_label) return true;
	else return false;
}

void NeuralNetwork::training(const vector<vector<float>>& train_images,
	const vector<unsigned char>& train_labels, const vector<vector<float>>& test_images,
	const vector<unsigned char>& test_labels, int num_epoch, float learning_rate)
{
	for (size_t i = 0; i < num_epoch; ++i) {
		cout << "Epoch " << i + 1 << endl;
		int count = 0;
		int size = 100;
		for (size_t j = 0; j < size/*train_images.size()*/; ++j) {
			auto& train_image = train_images[j];
			forward(train_image);

			vector<float> predict_label = layers.back()->getAllA();
			vector<unsigned char> true_label = createTrueLabelVector(train_labels[j]);
			/*if (j % 1000 == 0) {
				for (auto& value : predict_label) {
					cout << value << " ";
				}
				cout << endl;
				for (auto& value : true_label) {
					cout << float(value) << " ";
				}
				cout << endl;
			}*/
			backward(train_image, predict_label, true_label, learning_rate);
			/*if (j % 1000 == 0) {
				vector<float> output_weight = layers.back()->getNodes()[0].getWeight();
				for (size_t k = 0; k < output_weight.size(); ++k) {
					cout << output_weight[k] << " ";
				}
				cout << endl;
			}*/
		}

		for (size_t j = 0; j < size/*test_images.size()*/; ++j) {
			forward(test_images[j]);
			vector<float> predict_label = layers.back()->getAllA();
			if (getAccuracy(predict_label, test_labels[j])) {
				count++;
			}
		}

		float accuracy = static_cast<float>(count) / test_images.size() * 100.0f;
		cout << "Accuracy: " << accuracy << "%" << endl;
	}
}