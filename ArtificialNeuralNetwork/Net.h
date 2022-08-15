#pragma once
#include <vector>
#include "Neuron.h"
typedef std::vector<Neuron> Layer;

class Net
{
public:
	//Constructor
	Net(const std::vector<unsigned> &topology);

	//Interface functions
	void feedForward(const std::vector<double> &inputVals) {};
	void backProp(const std::vector<double> &targetVals) {};
	void getResults(std::vector<double> &resultVals) const {};
private:
	//2d vector of type layer
	std::vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
};

