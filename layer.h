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

    const vector<Node>& getNodes() {
        return nodes;
    }

    vector<float> getAllZ() {
		vector<float> vec;
		for (const auto& node : nodes) {
			vec.push_back(node.getZ());
		}
		return vec;
    }

	vector<float> getAllA() {
		vector<float> vec;
		for (const auto& node : nodes) {
			vec.push_back(node.getA());
		}
		return vec;
	}

    // Polymorphic interface
    virtual void forward(const vector<float>& inputs);
    
    virtual vector<float> backward(const vector<float>& dA, const vector<float>& pre_layer, float learning_rate);
};

class OutputLayer : public Layer {
public:
    OutputLayer(int num_nodes, int num_weights_per_node) : Layer(num_nodes, num_weights_per_node) {}

    virtual ~OutputLayer() = default;

    // Override to apply softmax activation
    void forward(const vector<float>& inputs) override;

	vector<float> backward(const vector<float>& dA, const vector<float>& pre_layer, float learning_rate) override;
};