#pragma once
#include <math.h>
#include <vector>

using namespace std;

class ActivationFunction
{
public:
    // ReLU activation
    static vector<float> relu(const vector<float> &input)
    {
        vector<float> output(input.size());
        for (size_t i = 0; i < input.size(); ++i)
        {
            output[i] = max(0.0f, input[i]);
        }
        return output;
    }

    // ReLU derivative
    static vector<float> relu_derivative(const vector<float> &input)
    {
        vector<float> output(input.size());
        for (size_t i = 0; i < input.size(); ++i)
        {
            output[i] = (input[i] > 0) ? 1.0f : 0.0f;
        }
        return output;
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
};