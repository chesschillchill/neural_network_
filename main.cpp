#include <iostream>

#include "neural_network.h"
int main() {
	NeuralNetwork nn(3, 5, 4, 2); // Example initialization with 3 inputs, 5 nodes in first hidden layer, 4 in second, and 2 outputs

	std::cout << "Neural Network initialized with 3 inputs, 5 hidden nodes in layer 1, 4 hidden nodes in layer 2, and 2 outputs." << std::endl;
	
	return 0;
}