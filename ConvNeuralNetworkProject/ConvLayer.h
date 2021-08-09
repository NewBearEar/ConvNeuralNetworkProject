#pragma once
#include "LayerBase.h"
#include "Util.h"


class ConvLayer :
    public LayerBase
{
public:
    ConvLayer(DataSize inputSize, Params hyperParams,double batch_m);

    virtual LayerType getLayerType() override;
    virtual shared_ptr<DataStruct> forward(DataStruct& A_prev) override;
    virtual shared_ptr<DataStruct> backward(DataStruct& dZ,double learningRate) override;
    DataSize computeOutputSize(DataSize inputSize, Params hyperParams);
    

    //¾í»ýºËÐÎ×´(c_n,height,weight,c_prev)
    shared_ptr<DataStruct> kernel;
    shared_ptr<DataStruct> kernelGradient;
    shared_ptr<DataStruct> bias;
    shared_ptr<DataStruct> biasGradient;
    shared_ptr<DataStruct> inputX;

    //m¸öimageCol
    vector<shared_ptr<mat>> imageCols;
};


