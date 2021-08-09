#pragma once
#include <vector>

//定义数据size类
using namespace std;
class DataSize
{
public:
	DataSize() = default;
	DataSize(int in_number,int in_height,int in_width ,int in_channel)
		:number(in_number), width(in_width),height(in_height),channel(in_channel){}
	//0 number ,1 height,2 width ,3 channel
	DataSize(vector<int> sizeVect);
	//get方法
	inline int getTotalSize() const { return get4DSize(); }
	inline int get4DSize() const { return number * channel * width * height; }
	inline int get3DSize() const { return channel * width * height; }
	inline int get2DSize() const { return width * height; }
	//0 number ,1 height,2 width ,3 channel
	vector<int> getSize();
	//set方法
	inline void setSize(const int in_number, const int in_height,
		const int in_width, const int in_channel){
		this->number = in_number;
		this->width = in_width;
		this->height = in_height;
		this->channel = in_channel;
	}
private:
	int number = 0;
	int width = 0;
	int height = 0;
	int channel = 0;
	vector<int> sizeVect;
};

