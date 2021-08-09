#pragma once

#include "DataSize.h"
#include <armadillo>

using namespace std;
using namespace arma;


//定义数据类型类，存储四维矩阵
class DataStruct
{
public:
	DataStruct() = default;
	DataStruct(DataSize &size);
	DataStruct(vector<shared_ptr<dcube>>& vecCube);
	DataStruct(DataSize &size, vector<shared_ptr<dcube>> &vecCube);
		
	virtual ~DataStruct();

	DataSize getSize() const;
	vector<shared_ptr<dcube>>& getData();

	void createZeroVecCubeBySize();
	void createRandnVecCubeBySize();

private:
	shared_ptr<DataSize> pSize;
	//通过vector与cube组合
	vector<shared_ptr<dcube>> vecCube;
};

