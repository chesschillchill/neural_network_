#pragma once
#include "node.h"
#include "activation_function.h"

class Layer {
protected:
    vector<Node> nodes;
public:
    Layer() = default;

    Layer(int num_nodes, int num_weights_per_node) {
        nodes.resize(num_nodes);
        for (int i = 0; i < num_nodes; ++i) {
            nodes[i] = Node(num_weights_per_node);
        }
    }

    virtual ~Layer() = default;

    void setNodeWeight(int node_index, int weight_index, float value) {
        if (node_index >= 0 && node_index < nodes.size()) {
            nodes[node_index].setWeight(weight_index, value);
        }
    }

    void setNodeBias(int node_index, float value) {
        if (node_index >= 0 && node_index < nodes.size()) {
            nodes[node_index].setBias(value);
        }
    }

    float getNodeBias(int node_index) const {
        if (node_index >= 0 && node_index < nodes.size()) {
            return nodes[node_index].getBias();
        }
        return 0.0f;
    }

    virtual const vector<Node>& getNodes() const {
        return nodes;
    }

    // Polymorphic interface
    virtual vector<float> calculateValues(const vector<float>& inputs) {
        vector<float> values(nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i) {
            values[i] = nodes[i].calculateValue(inputs);
        }
        return activationLayer(values);
    }

    virtual vector<float> activationLayer(const vector<float>& values) {
		return ActivationFunction::relu(values);
    }

    virtual vector<float> derActivationLayer(const vector<float>& values) {
        return ActivationFunction::reluDerivative(values);
    }
};

class OutputLayer : public Layer {
public:
    OutputLayer(int num_nodes, int num_weights_per_node) : Layer(num_nodes, num_weights_per_node) {}

    virtual ~OutputLayer() = default;

    // Override to apply softmax activation
    vector<float> calculateValues(const vector<float>& inputs) override {
        vector<float> values(nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i) {
            values[i] = nodes[i].calculateValue(inputs);
        }
        return activationLayer(values);
    }

    vector<float> activationLayer(const vector<float>& values) override {
        return ActivationFunction::softmax(values);
    }

    vector<float> derActivationLayer(const vector<float>& values) override {
        return ActivationFunction::softmaxDerivative(values);
    }
};