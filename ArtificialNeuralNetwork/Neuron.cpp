#include "Neuron.h"
#include<vector>
#include <random>
#include<cstdlib>
#include <cmath>

typedef std::vector<Neuron> Layer;

struct Connection {
	double weight;
	double deltaWeight;
};
Neuron::Neuron(unsigned numOutput, unsigned m_myIndex)
{
	for (int i = 0; i < numOutput; i++)
	{	
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}
	m_myIndex = m_myIndex;
}

void Neuron::feedForward(const Layer &prevLayer) {
	double sum = 0.0;

	//suming all the previous layer
	for (int n = 0; n < prevLayer.size(); n++)
	{
		sum += prevLayer[n].m_outputVal * prevLayer[n].m_outputWeights[m_myIndex].weight;
	}
	m_outputVal = Neuron::transferFunction(sum);
}

double Neuron::transferFunction(double x) 
{
	// we will use a tanh function for values within 0-1
	return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
	return (1-x*x);
}