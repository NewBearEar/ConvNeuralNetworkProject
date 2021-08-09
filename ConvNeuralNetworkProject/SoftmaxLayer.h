#pragma once
#include "LayerBase.h"
class SoftmaxLayer :
    public LayerBase
{
public:
    SoftmaxLayer() = default;
    virtual LayerType getLayerType() {
        return LayerType::SOFTMAX_LAYER;
    }
    mat calLoss(mat predict, mat label, double& loss);
    mat predict(mat predict);

    //batch_m , class
    mat softmax;
};

