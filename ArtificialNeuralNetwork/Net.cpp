#include "Net.h"
#include <iostream>
#include <vector>
#include <cassert>
#include "Neuron.h"
typedef std::vector<Neuron> Layer;

//Constructor
Net::Net(const std::vector<unsigned> &topology) 
{
	unsigned numLayers = topology.size();
	//loop to create all the layers
	for (int layerNum = 0; layerNum < numLayers; layerNum++)
	{
		m_layers.push_back(Layer());
		unsigned numOutput = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		//Loop to create all the neurons and adds an extra bias layer
		for (int neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++)
		{
			m_layers.back().push_back(Neuron(numOutput,neuronNum));
			std::cout << "Neuron created!!" << std::endl;
		}
		m_layers.back().back().setOutputVal(1.0);
	}
}

void Net::feedForward(const std::vector<double>& inputVals) {
	assert(inputVals.size() == m_layers[0].size() - 1);

	//Latching the input values into each input neuron
	for (int i = 0; i < inputVals.size(); i++)
	{
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	//forward propagation
	for (int layerNum = 1; layerNum < m_layers.size(); layerNum++)
	{
		Layer &prevLayer = m_layers[layerNum - 1];

		for (int neuronNum = 0; neuronNum < m_layers[layerNum].size() - 1; neuronNum++)
		{
			m_layers[layerNum][neuronNum].feedForward(prevLayer);
		}
	}
}

void Net::backProp(const std::vector<double>& targetVals)
{
	//calculating the overall error using root mean square
	Layer &outputLayer = m_layers.back();
	m_error = 0.0;

	for (int n = 0; n < outputLayer.size() - 1; n++)
	{
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += m_error * m_error;
	}
	m_error /= outputLayer.size() - 1;
	m_error = sqrt(m_error); //rms value

	//recent average measurement
	m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);

	//output layer gradient
	for (int n = 0; n < outputLayer.size() - 1; n++)
	{
		outputLayer[n].calculateOutputGradients(targetVals[n]);
	}

	//hidden layer gradient
	for (int layerNum = m_layers.size() - 2; layerNum>0; layerNum--)
	{
		Layer &hiddenLayer = m_layers[layerNum];
		Layer& nextLayer = m_layers[layerNum + 1];
		
		for (int n = 0; n < hiddenLayer.size(); n++)
		{
			hiddenLayer[n].calculateHiddenGradients(nextLayer);
		}
	}
	//update all weights from all layers
	for (int layerNum = m_layers.size() - 1; layerNum > 0; layerNum--)
	{
		Layer& layer = m_layers[layerNum];
		Layer& prevLayer = m_layers[layerNum - 1];

		for (int n = 0; m_layers.size() - 1; n++)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}

}

void Net::getResults(std::vector<double>& resultVals) const 
{
	resultVals.clear();
	
	for (int n = 0; n < m_layers.back().size(); n++)
	{
		Neuron ne = m_layers.back()[n];
		resultVals.push_back(ne.getOutputVal());
	}
}