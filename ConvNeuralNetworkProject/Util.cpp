#include "Util.h"

mat Util::img2col(const dcube x, int k_w,int k_h, int stride,int pad)
{
	//ksizeָ����˵ĳ���
	//imgָ��ɫͼcube��colָ�������չ��֮��ľ���(f_h*f_w , k_w*k_h*x_c)
	//���ڼ��پ������
	long x_h = x.n_rows;
	long x_w = x.n_cols;
	long x_c = x.n_slices;
	long f_w = floor((x_w + 2 * pad - k_w) / stride) + 1;
	long f_h = floor((x_h + 2 * pad - k_h) / stride) + 1;
	mat imageCol(f_w * f_h, k_w * k_h * x_c,fill::zeros);
	int count = 0;
	dcube reshapeCube;
	for (int i = 0; i < f_h; i++) {
		for (int j = 0; j < f_w; j++) {
			reshapeCube = reshape(x(span(i*stride,i*stride + k_h-1),span(j*stride,j*stride+k_w-1),span(0,x_c-1)),1, k_w * k_h * x_c,1);
			mat colMat = reshapeCube.slice(0);
			imageCol(count, span(0, k_w * k_h * x_c - 1)) = colMat;
			count++;
		}
	}
	//ע��չƽ˳������
	/*
	* [[0,1],
	*  [2,3]]
	* չƽΪ
	* [0,2,1,3]
	* �����Ҳ��Ҫ�������˳��[�У��У�ͨ��]
	*/
	return imageCol;
}

dcube Util::cubeZeroPadding(const dcube x,int pad)
{
	uword x_rows = x.n_rows;
	uword x_cols = x.n_cols;
	uword x_slices = x.n_slices;
	long xPad_rows = x_rows + long(2) * long(pad);
	long xPad_cols = x_cols + long(2) * long(pad);
	//����������������padding
	dcube xPad(xPad_rows, xPad_cols, x_slices,fill::zeros);
	xPad(span(pad, pad + x_rows - 1), span(pad, pad + x_cols - 1), span(0, x_slices - 1)) = x;
	return xPad;
}

DataStruct Util::tensorZeroPadding( DataStruct ds, int pad)
{
	vector<shared_ptr<dcube>> tensor;
	dcube padCube;
	for (int i = 0; i < ds.getData().size(); i++) {
		padCube = cubeZeroPadding(*(ds.getData()[i]), pad);
		shared_ptr<dcube> pPadCube = make_shared<dcube>(padCube);
		//pPadCube.reset(&padCube);
		tensor.push_back(pPadCube);
	}
	return DataStruct(tensor);
}

dcube Util::reshapeMat2Cube(const mat m, int cubeRow, int cubeCol, int cubeSlice)
{
	//sliceά��Ϊ1��cube����reshape��mat����֮����
	//��ˣ���Ҫ���armadillo�޷���mat reshape��cube������
	int matRow = m.n_rows;
	int matCol = m.n_cols;
	if (matRow * matCol != cubeRow * cubeCol * cubeSlice) {
		throw exception("Ԫ��������ƥ�䣬�޷�reshape");
	}
	dcube result(cubeRow, cubeCol, cubeSlice);
	int count = 0;
	//ע��ѭ��˳��
	for (int j = 0; j < matCol; j++) {
		for (int i = 0; i < matRow; i++) {
			result((count % (cubeRow * cubeCol)) % cubeRow, (count % (cubeRow * cubeCol)) / cubeRow, count / (cubeRow * cubeCol)) = 
				m(i,j);
			count++;
		}
	}

	return result;
}

mat Util::reshapeColvec2MatByRow(const mat colMat, int row, int col)
{
	//���뱣֤row =1
	int matRow = colMat.n_rows;
	int matCol = colMat.n_cols;
	if (matRow * matCol != row * col) {
		throw exception("Ԫ��������ƥ�䣬�޷�reshape");
	}
	mat result(row,col);
	int count = 0;
	//ע��ѭ��˳��
	for (int j = 0; j < matRow; j++) {
		for (int i = 0; i < matCol; i++) {
			result(count /col, count % col) = colMat(j, i);
			count++;
		}
	}

	return result;
}

dcube Util::cubeRot180(const dcube c, int axis1, int axis2)
{
	int nr = c.n_rows;
	int nc = c.n_cols;
	int ns = c.n_slices;
	dcube cRot180(nr, nc, ns);

	if ((axis1 == 0 && axis2 == 1) || (axis1 == 1 && axis2 == 0)) {
		for (int s = 0; s < ns; s++) {
			mat tempM = c.slice(s);
			mat tempMRot180 = matRot180(tempM);
			cRot180.slice(s) = tempMRot180;
		}
	}
	if ((axis1 == 1 && axis2 == 2) || (axis1 == 2 && axis2 == 1)) {
		for (int r = 0; r < nr; r++) {
			mat tempM = c.row(r);
			mat tempMRot180 = matRot180(tempM);
			cRot180.row(r) = tempMRot180;
		}
	}
	if ((axis1 == 0 && axis2 == 2) || (axis1 == 2 && axis2 == 0)) {
		for (int i = 0; i < nc; i++) {
			mat tempM = c.col(i);
			mat tempMRot180 = matRot180(tempM);
			cRot180.col(i) = tempMRot180;
		}
	}


	return cRot180;
}

mat Util::matRot180(const mat m)
{
	int nr = m.n_rows;
	int nc = m.n_cols;
	mat mRot180(nr,nc);
	//��ת����
	for (int j = nc-1; j >= 0; j--) {
		for (int i = nr-1; i >= 0; i--) {
			mRot180(nr - 1 - i, nc - 1 - j) = m(i, j);
		}
	}
	return mRot180;
}

mat Util::matRot180(const dcube c)
{	
	//����άΪ1��cube
	return mat();
}

vector<shared_ptr<dcube>> Util::subtract(vector<shared_ptr<dcube>> a, vector<shared_ptr<dcube>> b)
{
	//a��bͬ��ά��
	vector<shared_ptr<dcube>> result;
	dcube temp;
	for (int i = 0; i < a.size(); i++) {
		temp = (*(a[i])) - (*(b[i]));
		shared_ptr<dcube> p = make_shared<dcube>(temp);
		result.push_back(p);
	}
	return result;
}

vector<shared_ptr<dcube>> Util::multyply_num(vector<shared_ptr<dcube>> a, double num)
{
	vector<shared_ptr<dcube>> result;
	dcube temp;
	for (int i = 0; i < a.size(); i++) {
		temp = (*(a[i])) * num;
		shared_ptr<dcube> p = make_shared<dcube>(temp);
		result.push_back(p);
	}
	return result;
}

void Util::useMaskSelectBigger(dcube &input, dcube &output, ucube mask)
{

	for (int r = 0; r < mask.n_rows; r++) {
		for (int c = 0; c < mask.n_cols; c++) {
			for (int s = 0; s < mask.n_slices; s++) {
				if (mask(r, c, s) == 0) {
					output(r, c, s) = 0;
				}
				else {
					output(r, c, s) = input(r, c, s);
				}
			}
		}
	}

	return ;
}

void Util::repeatRowCol2(dcube& input, dcube& output)
{
	//��input ��row col �ֱ�repeat 2
	//input: h ,w, c
	//output: 2*h , 2*w, c
	int h = input.n_rows;
	int w = input.n_cols;
	int c = input.n_slices;

	for (int ci = 0; ci < c; ci++) {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				output(2 * i, 2 * j, ci) = output(2 * i + 1, 2 * j, ci) = 
					output(2 * i, 2 * j + 1, ci) = output(2 * i + 1, 2 * j + 1, ci) = input(i, j, ci);
			}
		}
	}
}

int Util::reverseInt(unsigned char* a)
{
	return ((((a[0] * 256) + a[1]) * 256) + a[2]) * 256 + a[3];
}

mat Util::readMnistImage(const char* fileName)
{
	FILE* File = fopen(fileName, "r");
	fseek(File, 0, 0);
	mat image;
	unsigned char a[4];
	fread(a, 4, 1, File);
	int magic = reverseInt(a);
	if (magic != 2051) //magic number wrong
	{
		cout << magic;
		return mat(0, 0, fill::zeros);
	}
	fread(a, 4, 1, File);
	const int num_img = reverseInt(a);
	fread(a, 4, 1, File);
	const int num_row = reverseInt(a);
	fread(a, 4, 1, File);
	const int num_col = reverseInt(a);
	const int size_img = num_col * num_row;
	// �ļ�ͷ��ȡ���
	image.reshape(num_img, size_img);
	unsigned char img[784];
	for (int i = 0; i < num_img; i++)
	{
		fseek(File, i * 784 + 16, SEEK_SET);
		fread(img, size_img, 1, File);
		for (int j = 0; j < size_img; j++)
		{
			image(i, j) = double(img[j])/256;
		}
	}
	fclose(File);
	return image;
}

mat Util::readMnistLabel(const char* fileName)
{
	FILE* File = fopen(fileName, "r");
	fseek(File, 0, 0);
	unsigned char a[4];
	fread(a, 4, 1, File);
	int magic = reverseInt(a);
	if (magic != 2049) //magic number wrong
	{
		cout << magic << " wrong magic";
		return mat(0, 0, fill::zeros);
	}
	fread(a, 4, 1, File);
	const int num_lab = reverseInt(a);
	// �ļ�ͷ��ȡ���
	mat label(num_lab, 10, fill::zeros);
	unsigned char lab[1];
	for (int i = 0; i < num_lab; i++)
	{
		fread(lab, 1, 1, File);
		label(i, int(lab[0])) = 1;
	}
	fclose(File);
	return label;
}


void Util::saveMat2csv(mat arr, char* filename)
{
	int row = arr.n_rows;
	int col = arr.n_cols;
	ofstream outFile;
	outFile.open(filename, ios::out); // ��ģʽ��ʡ��
	for (int j = 0; j < row; j++)
	{
		for (int i = 0; i < col; i++)
		{
			if (i < col - 1)
			{
				outFile << arr(j,i) << ",";
			}
			else
			{
				outFile << arr(j,i);
			}
		}
		outFile << '\n';
	}
	outFile.close();

}

vector<shared_ptr<dcube>> Util::cutTensorBatch(vector<shared_ptr<dcube>> org, int start, int end)
{
	vector<shared_ptr<dcube>> result;
	//dcube temp;
	for (int i = start; i < end; i++) {
		result.push_back(org[i]);
	}
	return result;
}


DataStruct Util::mnistImagePrepare(const char* fileName)
{
	// shape : m , n_w*n_h
	mat imageMat = Util::readMnistImage(fileName);
	dcube eachImageCube;
	vector<shared_ptr<dcube>> tensor;
	for (int i = 0; i < imageMat.n_rows; i++) {

		eachImageCube = Util::reshapeMat2Cube(Util::reshapeColvec2MatByRow(imageMat.row(i), 28, 28), 28, 28, 1);
		shared_ptr<dcube> p = make_shared<dcube>(eachImageCube);
		tensor.push_back(p);
	}
	DataStruct imageDS(tensor);
	return imageDS;
}

mat Util::mnistLabelPrepare(const char* fileName)
{
	// shape : m,class
	mat labelMat = Util::readMnistLabel(fileName);

	return labelMat;
}

void Util::saveDataStruct2MultyCubeFiles(DataStruct& ds, const char* dirName, const char* nameRoot)
{
	for (int i = 0; i < ds.getData().size(); i++) {
		string savePath = string(dirName) + "\\" + string(nameRoot) + to_string(i) + ".arma";
		(*(ds.getData()[i])).save(savePath);
	}
	
}

DataStruct Util::loadMultyCubeFiles2DataStruct(const char* dirName, const char* nameRoot, const int fileNum)
{
	
	vector<shared_ptr<dcube>> tempvec;
	dcube c;
	for (int i = 0; i < fileNum; i++) {
		string savePath = string(dirName) + "\\" + string(nameRoot) + to_string(i) + ".arma";
		c.load(savePath);
		shared_ptr<dcube> p = make_shared<dcube>(c);
		tempvec.push_back(p);
	}
	DataStruct ds(tempvec);
	return ds;
}

void Util::cv_mat_to_arma_mat(const cv::Mat1b& cv_mat_in, arma::uchar_mat& arma_mat_out)
{
	//convert unsigned char cv::Mat to arma::Mat<uchar>
	for (int r = 0; r < cv_mat_in.rows; r++) {
		for (int c = 0; c < cv_mat_in.cols; c++) {
			arma_mat_out(r, c) = cv_mat_in.data[r * cv_mat_in.cols + c];// / 255.0
		}
	}

}


