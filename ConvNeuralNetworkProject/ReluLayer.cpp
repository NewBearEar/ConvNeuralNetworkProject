#include "ReluLayer.h"
#include "Util.h"

LayerType ReluLayer::getLayerType()
{
	return LayerType::ACTIVATION_LAYER;
}

shared_ptr<DataStruct> ReluLayer::forward(DataStruct& A_prev)
{
	//cache : (A_prev ,W ,b) 
	shared_ptr<DataStruct> pA_prev = make_shared<DataStruct>(A_prev);
	this->cache.push_back(pA_prev);
	DataStruct reluA(A_prev.getSize());
	reluA.createZeroVecCubeBySize();
	dcube tempout;
	for (int i = 0; i < A_prev.getData().size(); i++) {
		ucube mask = (*(A_prev.getData()[i])) >= 0;
		tempout = dcube(*(A_prev.getData()[i]));
		Util::useMaskSelectBigger((*(A_prev.getData()[i])), tempout, mask);
		reluA.getData()[i] = make_shared<dcube>(tempout);
	}
	shared_ptr<DataStruct> A_output = make_shared<DataStruct>(reluA);
	return A_output;
}

shared_ptr<DataStruct> ReluLayer::backward(DataStruct& dZ)
{
	DataStruct x = *(this->cache[0]);
	DataStruct reluBack(dZ.getSize());
	reluBack.createZeroVecCubeBySize();
	dcube tempout;
	for (int i = 0; i < dZ.getData().size(); i++) {
		ucube mask = ((*(x.getData()[i])) >= 0);
		tempout = dcube(*(dZ.getData()[i]));
		Util::useMaskSelectBigger((*(dZ.getData()[i])), tempout, mask);
		reluBack.getData()[i] = make_shared<dcube>(tempout);
	}

	shared_ptr<DataStruct> dZ_output = make_shared<DataStruct>(reluBack);
	return dZ_output;
}
