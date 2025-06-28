#pragma once
#include <iostream>
#include <vector>
#include <random>

using std::vector;

class Node
{
private:
    // Weights connecting nodes of previous layer
	// First hidden layer connects to input layer
    vector<float> weights;
    float bias;

	// z = X * W + b
	// a = activation(vec_z)
    // softmax function required all nodes of layer
	float z = 0.0f;
	float a = 0.0f;
public:
    Node()
    {
        weights = vector<float>(0); // Initialize with no weights

        bias = 0.0f; // Initialize bias to 0
    }

    Node(int num_weights)
    {
        /*
        Generate random weights and bias for the node.
        Weights are initialized to random values between -1.0 and 1.0.
        Bias is also initialized to a random value in the same range.
        */

        weights.resize(num_weights);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
        for (int i = 0; i < num_weights; ++i)
        {
            weights[i] = dis(gen);
        }

        bias = dis(gen);

		// My neurals exploding so I change the initialization method to He initialization.
		// He initialization
		//weights.resize(num_weights);
		//std::random_device rd;
		//std::mt19937 gen(rd());
		//std::normal_distribution<float> dis(0.0f, sqrt(2.0f / num_weights)); 
		//for (int i = 0; i < num_weights; ++i)
		//{
		//	weights[i] = dis(gen);
		//}
		//bias = dis(gen); // Initialize bias with the same distribution
    }

    void setWeight(int index, float value)
    {
        if (index >= 0 && index < weights.size())
        {
            weights[index] = value;
        }
    }

    vector<float> getWeight() const
    {
		return weights;
    }

    void setBias(float value)
    {
        bias = value;
    }

    float getBias() const
    {
        return bias;
    }

	float getZ() const
	{
		return z;
	}

	void setZ(float value)
	{
		z = value;
	}

	float getA() const
	{
		return a;
	}

	void setA(float value)
	{
		a = value;
	}

    ~Node() = default;

    void forward(const vector<float> &inputs);

    void updateParameters(vector<float>& dWeights, float dBias, float learning_rate);
};