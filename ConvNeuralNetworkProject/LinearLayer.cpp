#include "LinearLayer.h"

LinearLayer::LinearLayer(int inChannel, int outChannel, int batch_m)
{
    this->inChannel = inChannel;
    this->outChannel = outChannel;
	double scale = sqrt(inChannel / 2.0);
    // weight shape : (inChannel,outChannel)
    mat weight(inChannel, outChannel, fill::randn);
    this->weight = weight / scale;

    // bias shape : (batch , outChannel)
    mat bias(batch_m, outChannel,fill::randn);
    this->bias = bias / scale;

    //初始化梯度
    mat weightGradient(inChannel, outChannel, fill::zeros);
    this->weightGradient = weightGradient;

    mat biasGradient(batch_m, outChannel, fill::zeros);
    this->biasGradient = biasGradient;

}

mat LinearLayer::forward(mat& x)
{
    // x shape : batch , inChannel  （其中inChannel为上一层图像三个维度乘积）
    this->x = x;
    mat xForward = this->x * this->weight + this->bias;
    return xForward;
}

mat LinearLayer::backward(mat& dZ, double learningRate)
{
    //dZ shape: batch , outChannel
    int outChannel = dZ.n_cols;
    int batch_m = this->x.n_rows;
    this->weightGradient = trans(this->x) * dZ / batch_m;
    mat bGradient(batch_m, outChannel);
    for (int i = 0; i < batch_m; i++) {
        bGradient.row(i) = sum(dZ, 0);
    }
    this->biasGradient = bGradient / batch_m;

    mat dZ_backward = dZ * trans(this->weight);
    //反向传播
    this->weight -= (this->weightGradient * learningRate);
    this->bias -= (this->biasGradient * learningRate);
    return dZ_backward;
}


