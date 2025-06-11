#include "layer.h"

vector<float> Layer::calculateValues(const vector<float> inputs) {
	vector<float> values;

	for (int i = 0; i < nodes.size(); ++i) {
		values.push_back(nodes[i].calculateValue(inputs));
	}

	return values;
}