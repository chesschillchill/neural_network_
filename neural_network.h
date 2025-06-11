#pragma once  
#include <iostream>  
#include <vector>  

#include "layer.h"  

using namespace std;  

class NeuralNetwork {  
private:  
	int imageSize;  
	int labelSize;  

	Layer hiddenLayer1;  
	Layer hiddenLayer2;  

	Layer outputLayer; // Assuming an output layer is needed, though not used in the constructor
public:  
	NeuralNetwork() {  
		imageSize = 0;
		labelSize = 0;
		hiddenLayer1 = Layer();  
		hiddenLayer2 = Layer();
		outputLayer = Layer();
	}  

	NeuralNetwork(int image_size, int num_hidden_nodes1, int num_hidden_nodes2, int label_size) {  
		imageSize = image_size;
		labelSize = label_size;  

		hiddenLayer1 = Layer(num_hidden_nodes1, image_size);
		hiddenLayer2 = Layer(num_hidden_nodes2, num_hidden_nodes1);  
		outputLayer = Layer(label_size, num_hidden_nodes2); 
	}  

	~NeuralNetwork() = default;  

	void forward(vector<vector<float>> train_image, vector<vector<int>> train_label);  
};
