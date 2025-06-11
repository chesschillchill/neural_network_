#pragma once
#include "node.h"

class Layer {
private:
	vector<Node> nodes;
public:
	Layer() = default; // Default constructor

	Layer(int num_nodes, int num_weights_per_node) {
		nodes.resize(num_nodes);

		for (int i = 0; i < num_nodes; ++i) {
			nodes[i] = Node(num_weights_per_node);
		}
	}
	void setNodeWeight(int node_index, int weight_index, float value) {
		if (node_index >= 0 && node_index < nodes.size()) {
			nodes[node_index].setWeight(weight_index, value);
		}
	}
	float getNodeWeight(int node_index, int weight_index) const {
		if (node_index >= 0 && node_index < nodes.size()) {
			return nodes[node_index].getWeight(weight_index);
		}
		return 0.0f;
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

	vector<float> calculateValues(const vector<float> inputs);

	const vector<Node>& getNodes() const {
		return nodes;
	}
	virtual ~Layer() = default;
};