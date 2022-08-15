#pragma once
#include "Neuron.h"

class Layer :
    public Neuron
{
public:
    Layer(Neuron neuron);
private:
    std::vector<Neuron> neuron_container;
};

