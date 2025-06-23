#include "node.h"

float Node::calculateValue(const vector<float> &inputs)
{
    /*
    node_ value = X * W + b
    */
    float value = 0.0f;

    if (inputs.size() != weights.size())
    {
        std::cout << "Error: Number of inputs does not match number of weights." << std::endl;
    }

    for (size_t i = 0; i < inputs.size(); ++i)
    {
        value += inputs[i] * weights[i];
    }

    value += bias;

    return value;
}