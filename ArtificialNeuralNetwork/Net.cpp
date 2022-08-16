#include "Net.h"
#include <iostream>
#include <vector>
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
			m_layers.back().push_back(Neuron(numOutput));
			std::cout << "Neuron created!!" << std::endl;
		}
	}
}