#pragma once
using namespace std;

//�洢�������������������״��pad��stride��ѧϰ��
class Params
{
public:
	//Ϊ�˴洢����
	//�������״(c_n,height,weight,c_prev)
	vector<int> kernelShape = {6,5,5,1};
	int pad = 0;
	int stride = 1;
	double learningRate = 0.1;
};

