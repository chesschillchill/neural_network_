#pragma once
#include <math.h>
#include <vector>

using namespace std;

class ActivationFunction
{
public:
	// Sigmoid activation
	static float sigmoid(float input)
	{
		return 1.0f / (1.0f + exp(-input));
	}

	// Sigmoid derivative
	static float sigmoidDerivative(float input)
	{
		float sig = sigmoid(input);
		return sig * (1.0f - sig);
	}

    // ReLU activation
	static float relu(float input)
	{
		return (input > 0) ? input : 0.0f;
	}

    // ReLU derivative
	static float reluDerivative(float input)
	{
		return (input > 0) ? 1.0f : 0.0f;
	}

    // Softmax
    static vector<float> softmax(const vector<float> &input)
    {
        vector<float> output(input.size());
        float sum = 0.0f;

        for (const auto &value : input)
        {
            sum += exp(value);
        }

        for (int i = 0; i < input.size(); ++i)
        {
            output[i] = exp(input[i]) / sum;
        }
        return output;
    }

	// Softmax derivative
	// GPT generated
	static vector<float> softmaxDerivative(const vector<float>& input)
	{
		vector<float> output(input.size());
		float sum = 0.0f;
		for (const auto& value : input)
		{
			sum += exp(value);
		}
		for (int i = 0; i < input.size(); ++i)
		{
			output[i] = exp(input[i]) / sum * (1 - exp(input[i]) / sum);
		}
		return output;
	}
};