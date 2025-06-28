#include "node.h"

void Node::forward(const vector<float>& inputs)
{
    /*
    node_ value = X * W + b
    */

    if (inputs.size() != weights.size())
    {
        std::cout << "Error: Number of inputs does not match number of weights." << std::endl;
        return;
    }

    for (size_t i = 0; i < inputs.size(); ++i)
    {
        z += inputs[i] * weights[i];
    }

    z += bias;
}
void Node::updateParameters(vector<float>& dWeights, float dBias, float learning_rate)
{
    for (size_t i = 0; i < weights.size(); ++i)
    {
        weights[i] -= learning_rate * dWeights[i];
    }
    bias -= learning_rate * dBias;
}
