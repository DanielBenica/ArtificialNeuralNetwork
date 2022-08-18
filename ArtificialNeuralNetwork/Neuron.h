#pragma once
#include <vector>
#include<random>
#include<cstdlib>
#include <cmath>
typedef std::vector<Neuron> Layer;

struct Connection {
	double weight;
	double deltaWeight;
};

class Neuron
{
public:
	Neuron(unsigned numOutput, unsigned m_myIndex);
	void setOutputVal(double val) { m_outputVal = val; };
	double getOutputVal(void) { return m_outputVal; };
	void feedForward(const Layer &prevLayer);
	void calculateOutputGradients(double targetVal);
	void calculateHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer& prevLayer);

private:
	static double eta;
	static double alpha;
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	std::vector<Connection> m_outputWeights;
	double m_outputVal;
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	unsigned m_myIndex;
	double m_gradient;
	double sumDOW(const Layer& nextLayer) const;
};

