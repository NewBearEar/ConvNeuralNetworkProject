#include "DataSize.h"

DataSize::DataSize(vector<int> sizeVect)
{
	number = sizeVect[0];
	height = sizeVect[1];
	width = sizeVect[2];
	channel = sizeVect[3];
}

vector<int> DataSize::getSize()
{
	sizeVect.clear();
	sizeVect.push_back(number);
	sizeVect.push_back(height);
	sizeVect.push_back(width);
	sizeVect.push_back(channel);
	return sizeVect;
	
}
