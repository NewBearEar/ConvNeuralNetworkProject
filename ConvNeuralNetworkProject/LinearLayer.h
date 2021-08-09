#pragma once
#include "LayerBase.h"
class LinearLayer :
    public LayerBase
{
public:
    LinearLayer(int inChannel, int outChannel,int batch_m);
    virtual LayerType getLayerType() {
        return LayerType::FC_LAYER;
    }
    virtual mat forward(mat& x) override;
    virtual mat backward(mat& dZ,double learningRate) override;
    
    mat x;
    mat weight;
    mat weightGradient;
    mat bias;
    mat biasGradient;
    int inChannel;
    int outChannel;
};

