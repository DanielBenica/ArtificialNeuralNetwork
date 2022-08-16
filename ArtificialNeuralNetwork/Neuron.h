#pragma once
#include <vector>

struct Connection {
	double weight;
	double deltaWeight;
};

class Neuron
{
public:
	Neuron(unsigned numOutput);
private:
	double m_outputVal;
	std::vector<Connection> m_outputWeights;
};

