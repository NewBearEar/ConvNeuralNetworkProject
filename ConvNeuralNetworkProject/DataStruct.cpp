#include "DataStruct.h"

DataStruct::DataStruct(DataSize& size)
{
	//使用reset将传统指针转为智能指针
	this->pSize = make_shared<DataSize>(size);
}

DataStruct::DataStruct(vector<shared_ptr<dcube>>& vecCube)
{
	
	this->vecCube = vecCube;
	this->pSize = make_shared<DataSize>();
	if (!this->vecCube.empty())
		this->pSize->setSize(vecCube.size(), vecCube[0]->n_rows,
			vecCube[0]->n_cols, vecCube[0]->n_slices);
			
}

DataStruct::DataStruct(DataSize& size, vector<shared_ptr<dcube>>& vecCube)
{
	this->pSize.reset(&size);
	this->vecCube = vecCube;
}

DataStruct::~DataStruct()
{
		

}

DataSize DataStruct::getSize() const
{
	return *(this->pSize);
}

vector<shared_ptr<dcube>>& DataStruct::getData()
{
	return this->vecCube;

}

void DataStruct::createZeroVecCubeBySize()
{
	this->vecCube.clear();
	//0 number ,1 height,2 width ,3 channel
	vector<int> sizeVec = this->pSize->getSize();
	dcube A;
	for (int i = 0; i < sizeVec[0]; i++) {
		A = dcube(sizeVec[1], sizeVec[2], sizeVec[3],fill::zeros);
		shared_ptr<dcube> pA = make_shared<dcube>(A);
		//pA.reset(&A);
		this->vecCube.push_back(pA);
	}
}

void DataStruct::createRandnVecCubeBySize()
{
	this->vecCube.clear();
	//0 number ,1 height,2 width ,3 channel
	vector<int> sizeVec = this->pSize->getSize();
	dcube A;
	for (int i = 0; i < sizeVec[0]; i++) {
		A = dcube(sizeVec[1], sizeVec[2], sizeVec[3], fill::randn);
		shared_ptr<dcube> pA = make_shared<dcube>(A);
		//pA.reset(&A);
		this->vecCube.push_back(pA);
	}
}
