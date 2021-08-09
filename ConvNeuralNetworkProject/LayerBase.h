#pragma once
#include "DataSize.h"
#include "DataStruct.h"
#include "Params.h"

using namespace std;

enum class LayerType {
	INPUT_LAYER,CONV_LAYER,ACTIVATION_LAYER,POOL_LAYER,FC_LAYER,SOFTMAX_LAYER
};
class LayerBase
{
	friend class Network;
public:
	virtual LayerType getLayerType() = 0;
	virtual shared_ptr<DataStruct> forward(DataStruct& A_prev, DataStruct& W, DataStruct& b);
	virtual shared_ptr<DataStruct> forward(DataStruct& A_prev);
	virtual mat forward(mat& x);
	virtual mat backward(mat& dZ,double learningRate);
	virtual shared_ptr<DataStruct> backward(DataStruct& dZ);
	virtual shared_ptr<DataStruct> backward(DataStruct& dZ, double learningRate);
	inline double getLearningRate() const { return learningRate; }
	inline void setLearningRate(double lr) { this->learningRate = lr; }
	inline void setLearningRateByParams() {
		this->learningRate = this->hyperParams.learningRate;
	}
	inline void setHyperParams(Params hParams) { this->hyperParams = hParams; }
	DataSize outputSize;
protected:
	//cache存储本层数据，用于反向传播
	//0 A_prev, 1 W , 2 b
	vector<shared_ptr<DataStruct>> cache;
	Params hyperParams;
	DataSize inputSize;
	
private:
	double learningRate = 0.1;

};


