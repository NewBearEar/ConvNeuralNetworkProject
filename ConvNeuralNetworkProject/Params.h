#pragma once
using namespace std;

//存储超参数，包括卷积核形状，pad，stride，学习率
class Params
{
public:
	//为了存储方便
	//卷积核形状(c_n,height,weight,c_prev)
	vector<int> kernelShape = {6,5,5,1};
	int pad = 0;
	int stride = 1;
	double learningRate = 0.1;
};

