#pragma once
#include "LayerBase.h"
class MaxPoolLyaer :
    public LayerBase
{
public:
    virtual LayerType getLayerType() {
        return LayerType::POOL_LAYER;
    }
    //ֻʵ��ksize=2��stride=2�����ػ�
    virtual shared_ptr<DataStruct> forward(DataStruct& A_prev);
    virtual shared_ptr<DataStruct> backward(DataStruct& dZ);
    shared_ptr<DataStruct> pfeatureMask;
};

