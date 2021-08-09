#include "SoftmaxLayer.h"

mat SoftmaxLayer::calLoss(mat predict, mat label, double& loss)
{
	int batch_m = predict.n_rows;
	int classes = predict.n_cols;
	this->predict(predict);
	loss = 0.0;
	//this->softmax = mat(predict.n_rows,predict.n_cols);
	mat delta(batch_m, classes, fill::zeros);
	for (int i = 0; i < batch_m; i++) {
		//cout << label.row(i);
		delta.row(i) = this->softmax.row(i) - label.row(i);
		//cout << this->softmax.row(i);
		//cout << log(this->softmax.row(i)) % label.row(i);
		mat summat = sum(log(this->softmax.row(i)) % label.row(i),1);
		loss -= summat(0, 0);
	}
	loss /= batch_m;
	return delta;
}

mat SoftmaxLayer::predict(mat predict)
{
	int batch_m = predict.n_rows;
	int classes = predict.n_cols;
	this->softmax = mat(batch_m, classes, fill::zeros);
	for (int i = 0; i < batch_m; i++) {
		mat predictTemp = predict.row(i) - max(predict.row(i));

		predictTemp = exp(predictTemp);
		//cout << "\n" << sum(predictTemp,1)<<"\n"<<predictTemp;
		mat summat = sum(predictTemp, 1);
		this->softmax.row(i) = predictTemp / summat(0,0);
		//cout << "\n" << this->softmax.row(i);
	}
	//cout << "\n" << this->softmax;
	return this->softmax;
}
