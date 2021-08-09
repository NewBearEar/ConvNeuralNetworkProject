#pragma once
#include "LayerBase.h"
class ReluLayer :
    public LayerBase
{
public:
    virtual LayerType getLayerType();
    virtual shared_ptr<DataStruct> forward(DataStruct& A_prev);
    virtual shared_ptr<DataStruct> backward(DataStruct& dZ);
};

