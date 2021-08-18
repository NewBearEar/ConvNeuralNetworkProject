#include "ConvNeuralNetworkWidget.h"
#include <armadillo>
#include "Util.h"
#include "ConvLayer.h"
#include "ReluLayer.h"
#include "MaxPoolLyaer.h"
#include "LinearLayer.h"
#include "SoftmaxLayer.h"
#include "opencv2/opencv.hpp"

#define CV_BGR2RGB 4

using namespace std;
using namespace arma;
ConvNeuralNetworkWidget::ConvNeuralNetworkWidget(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    connect(ui.pushButton, SIGNAL(clicked()), this, SLOT(clickBtnTrain()));
    connect(ui.pushButton_2, SIGNAL(clicked()), this, SLOT(clickBtnVal()));
    connect(ui.pushButton_3, SIGNAL(clicked()), this, SLOT(setTestImageNum()));
    connect(ui.pushButton_4, SIGNAL(clicked()), this, SLOT(predImageClass()));
    connect(this, SIGNAL(sendData(QString)), this, SLOT(updateTextEditData(QString)));
}

void ConvNeuralNetworkWidget::clickBtnTrain() {
    ui.textEdit->clear();
    nnTrain();
    return ; 
}



void ConvNeuralNetworkWidget::nnTrain()
{
    string datasetDirRoot = "..\\dataset\\mnist_dataset\\mnist_dataset\\";
    DataStruct train_input = Util::mnistImagePrepare((datasetDirRoot + string("train-images.idx3-ubyte")).c_str());
    
    mat train_label = Util::mnistLabelPrepare((datasetDirRoot + string("train-labels.idx1-ubyte")).c_str());
    
    Params hyperParams1;
    hyperParams1.kernelShape = { 6,5,5,1 };
    int pad = 0;
    int stride = 1;
    double learningRate = 0.05;
    hyperParams1.pad = 0;
    hyperParams1.stride = 1;
    hyperParams1.learningRate = learningRate;

    int batch = 32;
    DataSize inputBatchSize(batch, (train_input.getSize().getSize()[1]), (train_input.getSize().getSize()[2]),
        (train_input.getSize().getSize()[3]));
    //网络构建
    ConvLayer conv1 = ConvLayer(inputBatchSize, hyperParams1, batch);   // batch * 24 * 24 * 6
    ReluLayer relu1 = ReluLayer();
    MaxPoolLyaer pool1 = MaxPoolLyaer();                                // batch * 12 * 12 * 6
    // batch ,f_h/2,f_w/2 ,c_n
    DataSize layer1Outsize((conv1.outputSize.getSize())[0], (conv1.outputSize.getSize())[1] / 2,
        (conv1.outputSize.getSize())[2] / 2, (conv1.outputSize.getSize())[3]);

    Params hyperParams2(hyperParams1);
    hyperParams2.kernelShape = { 16,5,5,6 };
    ConvLayer conv2 = ConvLayer(layer1Outsize, hyperParams2, batch);            // batch * 8 * 8 * 16
    ReluLayer relu2 = ReluLayer();
    MaxPoolLyaer pool2 = MaxPoolLyaer();                                // batch * 4 * 4 * 16
    DataSize layer2Outsize((conv2.outputSize.getSize())[0], (conv2.outputSize.getSize())[1] / 2,
        (conv2.outputSize.getSize())[2] / 2, (conv2.outputSize.getSize())[3]);

    LinearLayer fc_nn = LinearLayer(layer2Outsize.get3DSize(), train_label.n_cols, layer2Outsize.getSize()[0]);
    SoftmaxLayer softmax = SoftmaxLayer();

    //参数保存路径
    const char* dirName = "..\\results";
    const char* conv1KernelRoot = "conv1_k";
    const char* conv1BiasRoot = "conv1_b";
    const char* conv2KernelRoot = "conv2_k";
    const char* conv2BiasRoot = "conv2_b";
    const char* fcWeightRoot = "fc_w";
    const char* fcBiasRoot = "fc_b";
    //迭代
    int epoches = 2;
    for (int epoch = 0; epoch < epoches; epoch++) {
        for (int i = 0; i < train_input.getSize().getSize()[0]; i += batch) {
            vector<shared_ptr<dcube>> vbatch = Util::cutTensorBatch(train_input.getData(), i, i + batch);
            DataStruct X(vbatch);
            mat Y = train_label(span(i, i + batch - 1), span(0, train_label.n_cols - 1));

            shared_ptr<DataStruct> predict = conv1.forward(X);
            predict = relu1.forward(*predict);
            //cout << *(predict->getData()[0]);
            //Util::saveDataStruct2MultyCubeCsv(*conv1.kernel,"D:\\ConvNNProj\\ConvNeuralNetworkProject\\results","conv1_k");
            predict = pool1.forward(*predict);
            //cout << *(predict->getData()[0]);
            predict = conv2.forward(*predict);
            //cout << *(predict->getData()[0]);
            predict = relu2.forward(*predict);
            //cout << *(predict->getData()[0]);
            predict = pool2.forward(*predict);
            //cout << *(predict->getData()[0]);
            mat flatPredict(batch, fc_nn.inChannel);
            for (int j = 0; j < batch; j++) {
                mat temp = reshape((*(predict->getData()[j])), 1, fc_nn.inChannel, 1);
                flatPredict.row(j) = temp;
            }
            mat fcPredict = fc_nn.forward(flatPredict);
            //cout << fcPredict;


            double loss = 0;
            mat delta = softmax.calLoss(fcPredict, Y, loss);
            //cout << delta;
            mat delta2 = fc_nn.backward(delta, learningRate);
            vector<shared_ptr<dcube>> vdZ;
            dcube tempcube;
            for (int j = 0; j < batch; j++) {
                tempcube = Util::reshapeMat2Cube(delta2.row(j), layer2Outsize.getSize()[1],
                    layer2Outsize.getSize()[2], layer2Outsize.getSize()[3]);
                shared_ptr<dcube> p = make_shared<dcube>(tempcube);
                vdZ.push_back(p);
            }
            DataStruct dZ(vdZ);
            shared_ptr<DataStruct> pdZ = pool2.backward(dZ);
            pdZ = relu2.backward(*pdZ);
            pdZ = conv2.backward(*pdZ, learningRate);
            pdZ = pool1.backward(*pdZ);
            pdZ = relu1.backward(*pdZ);
            conv1.backward(*pdZ, learningRate);
            if (i % 320 == 0) {
                string ep = string("Epoch-") + to_string(epoch) + string(": ") + to_string(i) + string("loss:{") + to_string(loss) + string("}\n");
                emit sendData(QString::fromStdString(ep));
                cout << ep;
            }
            //cout << "Epoch-" << epoch << ":" << i << ":" << "loss:{" << loss << "}\n";
            if (i % 20000 == 0)
                learningRate *= pow(0.9, (epoch + 1));
        }
        string f = "training finished ";
        emit sendData(QString::fromStdString(f));
        cout << f << "\n";
        learningRate *= pow(0.9, (epoch + 1));
        //每个epoch保存一下关键参数
        Util::saveDataStruct2MultyCubeFiles(*conv1.kernel, dirName, conv1KernelRoot);
        Util::saveDataStruct2MultyCubeFiles(*conv1.bias, dirName, conv1BiasRoot);
        Util::saveDataStruct2MultyCubeFiles(*conv2.kernel, dirName, conv2KernelRoot);
        Util::saveDataStruct2MultyCubeFiles(*conv2.bias, dirName, conv2BiasRoot);
        fc_nn.weight.save(string(dirName) + string("\\") + string(fcWeightRoot) + string(".arma"));
        fc_nn.bias.save(string(dirName) + string("\\") + string(fcBiasRoot) + string(".arma"));

    }
    

    //system("pause");
}

void ConvNeuralNetworkWidget::nnVal()
{
    //读取测试数据
    string datasetDirRoot = "..\\dataset\\mnist_dataset\\mnist_dataset\\";
    DataStruct test_input = Util::mnistImagePrepare((datasetDirRoot + string("t10k-images.idx3-ubyte")).c_str());
    mat test_label = Util::mnistLabelPrepare((datasetDirRoot + string("t10k-labels.idx1-ubyte")).c_str());

    //参数的保存路径
    const char* dirName = "..\\results";
    const char* conv1KernelRoot = "conv1_k";
    const char* conv1BiasRoot = "conv1_b";
    const char* conv2KernelRoot = "conv2_k";
    const char* conv2BiasRoot = "conv2_b";
    const char* fcWeightRoot = "fc_w";
    const char* fcBiasRoot = "fc_b";
    //读取参数
    int trainBatch = 32;
    int conv1KernelFile = 6;
    int conv2KernelFile = 16;
    int convBiasFile = trainBatch;
    DataStruct conv1_k = Util::loadMultyCubeFiles2DataStruct(dirName, conv1KernelRoot, conv1KernelFile);
    DataStruct conv1_b = Util::loadMultyCubeFiles2DataStruct(dirName, conv1BiasRoot, convBiasFile);
    DataStruct conv2_k = Util::loadMultyCubeFiles2DataStruct(dirName, conv2KernelRoot, conv2KernelFile);
    DataStruct conv2_b = Util::loadMultyCubeFiles2DataStruct(dirName, conv2BiasRoot, convBiasFile);
    mat fc_w;
    fc_w.load(string(dirName) + string("\\") + string(fcWeightRoot) + string(".arma"));
    mat fc_b;
    fc_b.load(string(dirName) + string("\\") + string(fcBiasRoot) + string(".arma"));

    int batch = 1;
    Params hyperParams1;
    hyperParams1.kernelShape = { 6,5,5,1 };
    int pad = 0;
    int stride = 1;
    double learningRate = 0.1;
    hyperParams1.pad = 0;
    hyperParams1.stride = 1;
    hyperParams1.learningRate = learningRate;
    //构建网络
    //挨个预测，相当于batch=1;
    DataSize inputBatchSize(batch, (test_input.getSize().getSize()[1]), (test_input.getSize().getSize()[2]),
        (test_input.getSize().getSize()[3]));
    //网络构建
    ConvLayer conv1 = ConvLayer(inputBatchSize, hyperParams1, batch);   // batch * 24 * 24 * 6
    ReluLayer relu1 = ReluLayer();
    MaxPoolLyaer pool1 = MaxPoolLyaer();                                // batch * 12 * 12 * 6
    // batch ,f_h/2,f_w/2 ,c_n
    DataSize layer1Outsize((conv1.outputSize.getSize())[0], (conv1.outputSize.getSize())[1] / 2,
        (conv1.outputSize.getSize())[2] / 2, (conv1.outputSize.getSize())[3]);

    Params hyperParams2(hyperParams1);
    hyperParams2.kernelShape = { 16,5,5,6 };
    ConvLayer conv2 = ConvLayer(layer1Outsize, hyperParams2, batch);            // batch * 8 * 8 * 16
    ReluLayer relu2 = ReluLayer();
    MaxPoolLyaer pool2 = MaxPoolLyaer();                                // batch * 4 * 4 * 16
    DataSize layer2Outsize((conv2.outputSize.getSize())[0], (conv2.outputSize.getSize())[1] / 2,
        (conv2.outputSize.getSize())[2] / 2, (conv2.outputSize.getSize())[3]);

    LinearLayer fc_nn = LinearLayer(layer2Outsize.get3DSize(), test_label.n_cols, layer2Outsize.getSize()[0]);
    SoftmaxLayer softmax = SoftmaxLayer();

    //将参数输入网络,需要注意由于C++不能自动广播bias，因此bias的维度与batch大小有关
    conv1.kernel = make_shared<DataStruct>(conv1_k);
    vector<shared_ptr<dcube>> con1BiasVec = Util::cutTensorBatch(conv1_b.getData(), 0, 0 + batch);
    DataStruct conv1Bias(con1BiasVec);
    conv1.bias = make_shared<DataStruct>(conv1Bias);
    conv2.kernel = make_shared<DataStruct>(conv2_k);
    vector<shared_ptr<dcube>> con2BiasVec = Util::cutTensorBatch(conv2_b.getData(), 0, 0 + batch);
    DataStruct conv2Bias(con2BiasVec);
    conv2.bias = make_shared<DataStruct>(conv2Bias);
    fc_nn.weight = fc_w;
    fc_nn.bias = fc_b.row(0);

    //计数器，统计准确率
    int num = 0;
    //10000个 ,batch=1
    for (int i = 0; i < test_input.getData().size(); i += batch) {
        vector<shared_ptr<dcube>> eachVec = Util::cutTensorBatch(test_input.getData(), i, i + batch);
        DataStruct X(eachVec);
        mat Y = test_label(i, span(0, test_label.n_cols - 1));

        shared_ptr<DataStruct> predict = conv1.forward(X);
        predict = relu1.forward(*predict);
        predict = pool1.forward(*predict);
        predict = conv2.forward(*predict);
        predict = relu2.forward(*predict);
        predict = pool2.forward(*predict);
        mat flatPredict(batch, fc_nn.inChannel);
        for (int j = 0; j < batch; j++) {
            mat temp = reshape((*(predict->getData()[j])), 1, fc_nn.inChannel, 1);
            flatPredict.row(j) = temp;
        }
        mat fcPredict = fc_nn.forward(flatPredict);

        mat softPredict = softmax.predict(fcPredict);
        if (softPredict.index_max() == Y.index_max()) {
            num++;
        }
        if (i % 100 == 0) {
            string ff = to_string(i) + string(" finished\n");
            emit sendData(QString::fromStdString(ff));
            cout << ff;
            //cout << i << " finished\n";
        }
    }
    string testacc = string("Test-ACC: ") + to_string((1.0 * num / test_input.getData().size()) * 100) + string("%\n");
    emit sendData(QString::fromStdString(testacc));
    cout << testacc;
    //cout << "Test-ACC: " << (1.0 * num / test_input.getData().size()) * 100 << "%\n";
    
}

int ConvNeuralNetworkWidget::nnPredict(int imageNumber)
{
    //读取测试数据
    string datasetDirRoot = "..\\dataset\\mnist_dataset\\mnist_dataset\\";
    DataStruct test_input = Util::mnistImagePrepare((datasetDirRoot + string("t10k-images.idx3-ubyte")).c_str());
    mat test_label = Util::mnistLabelPrepare((datasetDirRoot + string("t10k-labels.idx1-ubyte")).c_str());

    //参数的保存路径
    const char* dirName = "..\\results";
    const char* conv1KernelRoot = "conv1_k";
    const char* conv1BiasRoot = "conv1_b";
    const char* conv2KernelRoot = "conv2_k";
    const char* conv2BiasRoot = "conv2_b";
    const char* fcWeightRoot = "fc_w";
    const char* fcBiasRoot = "fc_b";
    //读取参数
    int trainBatch = 32;
    int conv1KernelFile = 6;
    int conv2KernelFile = 16;
    int convBiasFile = trainBatch;
    DataStruct conv1_k = Util::loadMultyCubeFiles2DataStruct(dirName, conv1KernelRoot, conv1KernelFile);
    DataStruct conv1_b = Util::loadMultyCubeFiles2DataStruct(dirName, conv1BiasRoot, convBiasFile);
    DataStruct conv2_k = Util::loadMultyCubeFiles2DataStruct(dirName, conv2KernelRoot, conv2KernelFile);
    DataStruct conv2_b = Util::loadMultyCubeFiles2DataStruct(dirName, conv2BiasRoot, convBiasFile);
    mat fc_w;
    fc_w.load(string(dirName) + string("\\") + string(fcWeightRoot) + string(".arma"));
    mat fc_b;
    fc_b.load(string(dirName) + string("\\") + string(fcBiasRoot) + string(".arma"));

    int batch = 1;
    Params hyperParams1;
    hyperParams1.kernelShape = { 6,5,5,1 };
    int pad = 0;
    int stride = 1;
    double learningRate = 0.1;
    hyperParams1.pad = 0;
    hyperParams1.stride = 1;
    hyperParams1.learningRate = learningRate;
    //构建网络
    //挨个预测，相当于batch=1;
    DataSize inputBatchSize(batch, (test_input.getSize().getSize()[1]), (test_input.getSize().getSize()[2]),
        (test_input.getSize().getSize()[3]));
    //网络构建
    ConvLayer conv1 = ConvLayer(inputBatchSize, hyperParams1, batch);   // batch * 24 * 24 * 6
    ReluLayer relu1 = ReluLayer();
    MaxPoolLyaer pool1 = MaxPoolLyaer();                                // batch * 12 * 12 * 6
    // batch ,f_h/2,f_w/2 ,c_n
    DataSize layer1Outsize((conv1.outputSize.getSize())[0], (conv1.outputSize.getSize())[1] / 2,
        (conv1.outputSize.getSize())[2] / 2, (conv1.outputSize.getSize())[3]);

    Params hyperParams2(hyperParams1);
    hyperParams2.kernelShape = { 16,5,5,6 };
    ConvLayer conv2 = ConvLayer(layer1Outsize, hyperParams2, batch);            // batch * 8 * 8 * 16
    ReluLayer relu2 = ReluLayer();
    MaxPoolLyaer pool2 = MaxPoolLyaer();                                // batch * 4 * 4 * 16
    DataSize layer2Outsize((conv2.outputSize.getSize())[0], (conv2.outputSize.getSize())[1] / 2,
        (conv2.outputSize.getSize())[2] / 2, (conv2.outputSize.getSize())[3]);

    LinearLayer fc_nn = LinearLayer(layer2Outsize.get3DSize(), test_label.n_cols, layer2Outsize.getSize()[0]);
    SoftmaxLayer softmax = SoftmaxLayer();

    //将参数输入网络,需要注意由于C++不能自动广播bias，因此bias的维度与batch大小有关
    conv1.kernel = make_shared<DataStruct>(conv1_k);
    vector<shared_ptr<dcube>> con1BiasVec = Util::cutTensorBatch(conv1_b.getData(), 0, 0 + batch);
    DataStruct conv1Bias(con1BiasVec);
    conv1.bias = make_shared<DataStruct>(conv1Bias);
    conv2.kernel = make_shared<DataStruct>(conv2_k);
    vector<shared_ptr<dcube>> con2BiasVec = Util::cutTensorBatch(conv2_b.getData(), 0, 0 + batch);
    DataStruct conv2Bias(con2BiasVec);
    conv2.bias = make_shared<DataStruct>(conv2Bias);
    fc_nn.weight = fc_w;
    fc_nn.bias = fc_b.row(0);

    //预测测试集中第i个图片的类别
    int i = imageNumber;
    vector<shared_ptr<dcube>> eachVec = Util::cutTensorBatch(test_input.getData(), i, i + batch);
    DataStruct X(eachVec);
    mat Y = test_label(i, span(0, test_label.n_cols - 1));

    shared_ptr<DataStruct> predict = conv1.forward(X);
    predict = relu1.forward(*predict);
    predict = pool1.forward(*predict);
    predict = conv2.forward(*predict);
    predict = relu2.forward(*predict);
    predict = pool2.forward(*predict);
    mat flatPredict(batch, fc_nn.inChannel);
    for (int j = 0; j < batch; j++) {
        mat temp = reshape((*(predict->getData()[j])), 1, fc_nn.inChannel, 1);
        flatPredict.row(j) = temp;
    }
    mat fcPredict = fc_nn.forward(flatPredict);

    mat softPredict = softmax.predict(fcPredict);
    int imgClass = softPredict.index_max();

    return imgClass;
}

void ConvNeuralNetworkWidget::showImageInLabel(int imageNumber)
{
    
    string datasetDirRoot = "..\\dataset\\mnist_dataset\\mnist_dataset\\";
    DataStruct test_input = Util::mnistImagePrepare((datasetDirRoot + string("t10k-images.idx3-ubyte")).c_str());
    mat test_label = Util::mnistLabelPrepare((datasetDirRoot + string("t10k-labels.idx1-ubyte")).c_str());
    //首先要将armadillo mat转cv Mat
    mat image1 = (*(test_input.getData()[imageNumber]))*256;
    cv::Mat_<double> c;
    Util::arma_mat_to_cv_mat(image1, c);
    //由于QImage限制矩阵必须为uchar类型，必须对double的矩阵强制转换
    cv::Mat_<uchar> cvMat = c;
    //openCV中的图像主要存储在Mat类中，要让其显示在Qt的Label控件上，必须先将其转换为Qt的QImage类
    //openCV使用的图像通道是BGR的而QImage使用的图像通道的RGB的。但这里第三维为1所以不用考虑
    cv::Mat rgb;
    QImage img;
    //cv::imshow("123", cvMat);
    if (cvMat.channels() == 3)
    {
        cv::cvtColor(cvMat, rgb,CV_BGR2RGB);
        img = QImage((const uchar*)(rgb.data), rgb.cols, rgb.rows, rgb.cols * rgb.channels(), QImage::Format_RGB888);
    }
    else
    {
        qDebug() << cvMat.cols << cvMat.rows << cvMat.step;
        img = QImage(cvMat.data, cvMat.cols, cvMat.rows,cvMat.step, QImage::Format_Indexed8);
        
    }

    //启用等比例缩放
    ui.label_2->setScaledContents(true);
    QSize qs = ui.label_2->rect().size();
    ui.label_2->setPixmap(QPixmap::fromImage(img).scaled(qs));
    //ui.label_2->resize(ui.label_2->pixmap()->size());
    ui.label_2->show();
    qApp->processEvents();

}

void ConvNeuralNetworkWidget::clickBtnVal()
{
    ui.textEdit->clear();
    nnVal();
}

void ConvNeuralNetworkWidget::setTestImageNum()
{
    QString num = ui.lineEdit->text();
    if (num.toInt()>=0 && num.toInt() < 10000 && !num.isEmpty()) {
        this->testImageNumber = num.toInt();
        showImageInLabel(this->testImageNumber);
    }
    else {
        QMessageBox::warning(this, "Out of Range!", "The number you entered should be one from 0 to 9999. Please enter a new one !");
    }
    
}

void ConvNeuralNetworkWidget::updateTextEditData(QString data)
{
    ui.textEdit->append(data);
    //实时刷新
    qApp->processEvents();
}

void ConvNeuralNetworkWidget::predImageClass()
{
    int imageClass = nnPredict(this->testImageNumber);
    QString predResult = QString::fromStdString((string("Number is : ") + to_string(imageClass)));
    //emit sendImageResult(predResult);
    ui.lineEdit_2->setText(predResult);
}

