#pragma once
#include <iostream>
#include <vector>
#include <random>

using std::vector;

class Node {
private:
	vector<float> weights;
	float bias;
public:
	Node()
	{
		weights = vector<float>(0); // Initialize with no weights

		bias = 0.0f; // Initialize bias to 0
	}

	Node(int num_weights){
		/*
		Generate random weights and bias for the node.
		Weights are initialized to random values between -1.0 and 1.0.
		Bias is also initialized to a random value in the same range.
		*/

		weights.resize(num_weights);
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
		for (int i = 0; i < num_weights; ++i) {
			weights[i] = dis(gen);
		}

		bias = dis(gen);
	}

	void setWeight(int index, float value) {
		if (index >= 0 && index < weights.size()) {
			weights[index] = value;
		}
	}

	float getWeight(int index) const {
		if (index >= 0 && index < weights.size()) {
			return weights[index];
		}
		return 0.0f;
	}

	void setBias(float value) {
		bias = value;
	}

	float getBias() const {
		return bias;
	}

	~Node() = default;

	float calculateValue(const vector<float>& inputs);
};