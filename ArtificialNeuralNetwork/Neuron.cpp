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
void Neuron::calculateOutputGradients(double targetVal)
{
	double delta = targetVal - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calculateHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::sumDOW(const Layer& nextLayer) const
{
	double sum = 0.0;

	//sum of the errors of the nodes
	for (int n = 0; n < nextLayer.size() - 1; n++)
	{
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}
	return sum;
}

void Neuron::updateInputWeights(Layer& prevLayer)
{
	for (int n = 0; n < prevLayer.size(); n++)
	{
		Neuron& neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
		double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient + alpha*oldDeltaWeight;

		neuron.m_outputWeights[n].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[n].weight += newDeltaWeight;
	}
}

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;