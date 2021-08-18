#pragma once
#include <armadillo>
#include "DataStruct.h"
#include "opencv2/opencv.hpp"
//#define uchar unsigned char

using namespace arma;

class Util
{
public:
	static mat img2col(const dcube x, int k_w, int k_h, int stride,int pad);
	static dcube cubeZeroPadding(const dcube x, int pad);
	static DataStruct tensorZeroPadding(DataStruct ds, int pad);
	static dcube reshapeMat2Cube(const mat m,int cubeRow,int cubeCol ,int cubeSlice);
	static mat reshapeColvec2MatByRow(const mat colMat, int row, int col);
	//这里对cube的操作太折磨了，直接写死，逆时针旋转一次180度
	static dcube cubeRot180(const dcube c, int axis1, int axis2);
	static mat matRot180(const mat m);
	static mat matRot180(const dcube c);

	static vector<shared_ptr<dcube>> subtract(vector<shared_ptr<dcube>> a, vector<shared_ptr<dcube>> b);
	static vector<shared_ptr<dcube>> multyply_num(vector < shared_ptr<dcube>> a, double num);
	static void useMaskSelectBigger(dcube& input, dcube& output, ucube mask);

	static void repeatRowCol2(dcube& input, dcube& output);
	//小端存储转换
	static int reverseInt(unsigned char* a);
	//读取image数据集信息
	static mat readMnistImage(const char* fileName);

	//读取label数据集信息
	static mat readMnistLabel(const char* fileName);

	static void saveMat2csv(mat arr, char* filename);
	static vector<shared_ptr<dcube>> cutTensorBatch(vector<shared_ptr<dcube>> org, int start, int end);

	static DataStruct mnistImagePrepare(const char* fileName);
	static mat mnistLabelPrepare(const char* fileName);
	static void saveDataStruct2MultyCubeFiles(DataStruct& ds, const char* dirName, const char* nameRoot);
	static DataStruct loadMultyCubeFiles2DataStruct(const char* dirName, const char* nameRoot, const int fileNum);
	
	static void cv_mat_to_arma_mat(const cv::Mat1b& cv_mat_in, arma::uchar_mat& arma_mat_out);
	template<typename T>
	inline static void arma_mat_to_cv_mat(const arma::Mat<T>& arma_mat_in, cv::Mat_<T>& cv_mat_out)
	{
		cv::transpose(cv::Mat_<T>(static_cast<int>(arma_mat_in.n_cols),
			static_cast<unsigned char>(arma_mat_in.n_rows),
			const_cast<T*>(arma_mat_in.memptr())),
			cv_mat_out);
	}

};


