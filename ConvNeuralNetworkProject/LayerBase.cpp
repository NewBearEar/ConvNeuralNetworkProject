#include "LayerBase.h"

shared_ptr<DataStruct> LayerBase::forward(DataStruct& A_prev, DataStruct& W, DataStruct& b)
{
	return shared_ptr<DataStruct>();
}

shared_ptr<DataStruct> LayerBase::forward(DataStruct& A_prev)
{
	return shared_ptr<DataStruct>();
}

mat LayerBase::forward(mat& x)
{
	return mat();
}

mat LayerBase::backward(mat& dZ, double learningRate)
{
	return mat();
}

shared_ptr<DataStruct> LayerBase::backward(DataStruct& dZ)
{
	return shared_ptr<DataStruct>();
}

shared_ptr<DataStruct> LayerBase::backward(DataStruct& dZ, double learningRate)
{
	return shared_ptr<DataStruct>();
}
