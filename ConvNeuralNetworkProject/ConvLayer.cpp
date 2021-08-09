#include "ConvLayer.h"

using namespace arma;
ConvLayer::ConvLayer(DataSize inputSize, Params hyperParams,double batch_m)
{
    this->inputSize = inputSize;
    this->hyperParams = hyperParams;
    DataSize outputSize = this->computeOutputSize(inputSize, hyperParams);
    this->outputSize = outputSize;
    //�������״(c_n,height,weight,c_prev)
    int kw = hyperParams.kernelShape[2];
    int kh = hyperParams.kernelShape[1];
    int kc_prev = hyperParams.kernelShape[3];
    int kc_n = hyperParams.kernelShape[0];
    //�Զ���batch��mini-batch����ʱӦ����m��ͬ
    double batch = batch_m;
    //scale
    double scale = sqrt(batch * kc_prev * kw * kh / kc_n);

    //��ʼ��kernel
    DataSize kernelSize(hyperParams.kernelShape);
    this->kernel = make_shared<DataStruct>(kernelSize);
    this->kernel->createRandnVecCubeBySize();
    for (int i = 0; i < this->kernel->getData().size(); i++) {
        (*(this->kernel->getData()[i])) /= scale;
    }
    //��ʼ��kernelGradient
    this->kernelGradient = make_shared<DataStruct>(kernelSize);
    this->kernelGradient->createZeroVecCubeBySize();
    //��ʼ��bias
    DataSize biasSize(outputSize);
    this->bias = make_shared<DataStruct>(biasSize);
    this->bias->createRandnVecCubeBySize();
    for (int i = 0; i < this->bias->getData().size(); i++) {
        (*(this->bias->getData()[i])) /= scale;
    }
    //��ʼ��biasGradient
    this->biasGradient = make_shared<DataStruct>(biasSize);
    this->biasGradient->createZeroVecCubeBySize();


}

LayerType ConvLayer::getLayerType()
{
    return LayerType::CONV_LAYER;
}

shared_ptr<DataStruct> ConvLayer::forward(DataStruct& A_prev)
{
    DataStruct padA_prev;
    shared_ptr<DataStruct> pA_prev;
    if (this->hyperParams.pad != 0) {
        padA_prev = Util::tensorZeroPadding(A_prev, this->hyperParams.pad);
        this->inputSize = pA_prev->getSize();
    }
    else {
        padA_prev = A_prev;
    }
    //�洢��һ������ݣ����ڷ��򴫲�
    pA_prev = make_shared<DataStruct>(padA_prev);
    this->cache.push_back(pA_prev);
    //����ͼ�����size
    DataSize featureSize = this->computeOutputSize(this->inputSize, this->hyperParams);
    this->outputSize = featureSize;
    //(m,f_h,f_w,n_C)
    shared_ptr<DataStruct> feature = make_shared<DataStruct>(featureSize);
    feature->createZeroVecCubeBySize();
    //�������ת����˷�����
    //չƽ�����(-1,nk)
    //�����ά�� c_n , k_h , k_w ,c_prev
    vector<int> kernelTensorShape = hyperParams.kernelShape;
    mat flatKernel(this->kernel->getSize().get3DSize(), kernelTensorShape[0]);
    for (int i = 0; i < kernelTensorShape[0]; i++) {
        mat tempM = reshape(*(this->kernel->getData()[i]), this->kernel->getSize().get3DSize(), 1, 1);
        flatKernel.col(i) = tempM;
    }
    //�������ͼ��ά�ȣ�m, f_h ,f_w ,c_n
    vector<int> outputShape = this->outputSize.getSize();
    //��ʼ���
    mat imageCol;
    for (int i = 0; i < pA_prev->getData().size(); i++) {
        imageCol = Util::img2col(*(pA_prev->getData()[i]), kernelTensorShape[2], kernelTensorShape[1],
            this->hyperParams.stride,this->hyperParams.pad);
        //����ͼ
        mat temp = imageCol * flatKernel;
        *(feature->getData()[i]) = Util::reshapeMat2Cube(temp, outputShape[1], outputShape[2], outputShape[3]);
        *(feature->getData()[i]) += *(this->bias->getData()[i]);

        shared_ptr<mat> pImageCol = make_shared<mat>(imageCol);
        //pImageCol.reset(&imageCol);
        this->imageCols.push_back(pImageCol);

    }
    //Util::img2col()
    this->cache.push_back(this->kernel);
    this->cache.push_back(this->bias);
        
    return feature;
}

shared_ptr<DataStruct> ConvLayer::backward(DataStruct& dZ,double learningRate)
{
    //��������x��ά�� m , n_h , n_w , c_prev
    vector<int> inputTensorShape = this->cache[0]->getSize().getSize();
    //�����Ȩ��k(w)��ά�� c_n ,k_h ,k_w ,c_prev
    vector<int> kernelTensorShape = this->kernel->getSize().getSize();
    //delta Z��ά���������򴫲�ʱ���ά����ͬ m, f_h , f_w ,c_n
    vector<int> deltaZTensorShape = dZ.getSize().getSize();

    this->hyperParams.learningRate = learningRate;
    //�������˺�bias���ݶ�
    cube dZCol(deltaZTensorShape[0],deltaZTensorShape[1]*deltaZTensorShape[2],deltaZTensorShape[3]);
    for (int i = 0; i < deltaZTensorShape[0]; i++) {
        mat tempM = reshape(*(dZ.getData()[i]), 1, deltaZTensorShape[1] * deltaZTensorShape[2], deltaZTensorShape[3]);
        dZCol.row(i)= tempM;         
    }
    mat kGradientTemp();
    //������ݶ�
    // totalKG shape : (kh*kw*c_prev ,c_n)
    mat totalKG(kernelTensorShape[1] * kernelTensorShape[2] * kernelTensorShape[3], kernelTensorShape[0],fill::zeros);
    for (int j = 0; j < deltaZTensorShape[0]; j++) {
        mat dZColEachM = dZCol.row(j);
        mat tempM = trans(*(this->imageCols[j])) * dZColEachM(span(0, deltaZTensorShape[1] * deltaZTensorShape[2] - 1),
            span(0, deltaZTensorShape[3] - 1));
        totalKG += tempM;
    }
    //totalKG = totalKG / deltaZTensorShape[0];
    //ת��Ϊvector<shared_ptr<dcube>>
    //vector<shared_ptr<dcube>> tensorKGVec;
    dcube tempTensorKG;
    for (int j = 0; j < kernelTensorShape[0]; j++) {
        mat tempC = totalKG.col(j);
        tempTensorKG = Util::reshapeMat2Cube(tempC, kernelTensorShape[1], kernelTensorShape[2], kernelTensorShape[3]);
        *(this->kernelGradient->getData()[j]) += tempTensorKG;

        *(this->kernelGradient->getData()[j]) /= deltaZTensorShape[0];
    }
    

    //DataStruct kGradient(tensorKGVec);
    //this->kernelGradient = make_shared<DataStruct>(kGradient);
    //bias�ݶ� 
    //tempBG shape: (1,1,c_n)
    dcube tempBG = sum(sum(dZCol, 0),1);
    //�ֶ��㲥 -> (m,f_h,f_w,c_n)
    //vector<shared_ptr<dcube>> tempTensorBGVec;
    dcube temp(deltaZTensorShape[1], deltaZTensorShape[2], deltaZTensorShape[3]);
    for (int j = 0; j < deltaZTensorShape[3]; j++) {
        dcube a(deltaZTensorShape[1], deltaZTensorShape[2], 1, fill::value(tempBG(0, 0, j)));
        temp.slice(j) = a;
    }
    for (int i = 0; i < deltaZTensorShape[0]; i++) {
        //shared_ptr<dcube> pTemp = make_shared<dcube>(temp);
        //tempTensorBGVec.push_back(pTemp);
        *(this->biasGradient->getData()[i]) += temp;

        *(this->biasGradient->getData()[i]) /= deltaZTensorShape[0];
    }

    DataStruct deltaBackwardDS(this->cache[0]->getSize());
    deltaBackwardDS.createZeroVecCubeBySize();
    //��ת�����
    vector<shared_ptr<dcube>> kernelRot180Vec;
    dcube kernelRot180;
    for (int i = 0; i < kernelTensorShape[0]; i++) {
        dcube temp = *(this->kernel->getData()[i]);
        kernelRot180 = Util::cubeRot180(temp, 0, 1);
        shared_ptr<dcube> pKernelRot180 = make_shared<dcube>(kernelRot180);
        kernelRot180Vec.push_back(pKernelRot180);
    }
    //����n_c��n_prevά ��0��3
    vector<shared_ptr<dcube>> kernelRot180SwapVec;
    dcube kernelRot180Swap(kernelTensorShape[1],kernelTensorShape[2],kernelTensorShape[0]);
    for (int j = 0; j < kernelTensorShape[3]; j++) {
        for (int i = 0; i < kernelTensorShape[0]; i++) {
            kernelRot180Swap.slice(i) = (*(kernelRot180Vec[i])).slice(j);
        }
        shared_ptr<dcube> p = make_shared<dcube>(kernelRot180Swap);
        // (n_prev ,k_h,k_w,c_n)
        kernelRot180SwapVec.push_back(p);
    }
    //reshape
    mat k180col(kernelTensorShape[0] * kernelTensorShape[1] * kernelTensorShape[2], kernelTensorShape[3]);
    for (int i = 0; i < kernelTensorShape[3]; i++) {
        mat temp = reshape((*(kernelRot180SwapVec[i])), kernelTensorShape[0] * kernelTensorShape[1] * kernelTensorShape[2], 1,1);
        k180col.col(i) = temp;
    }

    vector<shared_ptr<dcube>> pad_dZvec;
    dcube pad_dZ;
    if ((deltaZTensorShape[1] - kernelTensorShape[1] + 1) != inputTensorShape[1]) {
        int pad = floor((inputTensorShape[1] - deltaZTensorShape[1] + kernelTensorShape[1] - 1) / 2);
        for (int i = 0; i < deltaZTensorShape[0]; i++) {
            pad_dZ = Util::cubeZeroPadding((*(dZ.getData()[i])), pad);
            shared_ptr<dcube> p = make_shared<dcube>(pad_dZ);
            pad_dZvec.push_back(p);
        }
    }
    else {
        pad_dZvec = dZ.getData();
    }
    //���
    vector<shared_ptr<dcube>> dZ_backwardVec;
    dcube dZ_backward;
    for (int i = 0; i < inputTensorShape[0]; i++) {
        mat pad_dZCol = Util::img2col(*(pad_dZvec[i]), kernelTensorShape[2], kernelTensorShape[1],
            this->hyperParams.stride, this->hyperParams.pad);
        dZ_backward = Util::reshapeMat2Cube(pad_dZCol * k180col,inputTensorShape[1],
            inputTensorShape[2],inputTensorShape[3]);
        shared_ptr<dcube> pdz_backward = make_shared<dcube>(dZ_backward);
        dZ_backwardVec.push_back(pdz_backward);
    }
    DataStruct dZBackwardDS(dZ_backwardVec);
    shared_ptr<DataStruct> pdZBackwardDS = make_shared<DataStruct>(dZBackwardDS);

    //���򴫲�
    this->kernel->getData() = Util::subtract(this->kernel->getData(),
        Util::multyply_num(this->kernelGradient->getData(), this->hyperParams.learningRate));
    this->bias->getData() = Util::subtract(this->bias->getData(),
        Util::multyply_num(this->biasGradient->getData(), this->hyperParams.learningRate));
    return pdZBackwardDS;
}

DataSize ConvLayer::computeOutputSize(DataSize inputSize, Params hyperParams)
{
    //��������ά�� m , n_h , n_w , c_prev
    vector<int> inputTensorShape = inputSize.getSize();
    //�����ά�� c_n , k_h , k_w ,c_prev
    vector<int> kernelTensorShape = hyperParams.kernelShape;
    //padding and stride
    int p = hyperParams.pad;
    int s = hyperParams.stride;
    //��������ͼά��
    int f_h = floor((inputTensorShape[1] + 2.0 * p - kernelTensorShape[1]) / s) + 1;
    int f_w = floor((inputTensorShape[2] + 2.0 * p - kernelTensorShape[2]) / s) + 1;

    return DataSize(inputTensorShape[0],f_h,f_w,kernelTensorShape[0]);
}
