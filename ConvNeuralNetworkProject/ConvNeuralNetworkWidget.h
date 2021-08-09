#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_ConvNeuralNetworkWidget.h"
#include "gdal_priv.h"
#include <iostream>  
#include <QDebug>
#include <qmessagebox.h>


class ConvNeuralNetworkWidget : public QMainWindow
{
    Q_OBJECT

public:
    ConvNeuralNetworkWidget(QWidget *parent = Q_NULLPTR);

    //Õ¯¬Á—µ¡∑
    void nnTrain();
    //∆¿π¿
    void nnVal();
    //‘§≤‚“ª’≈Õº∆¨
    int nnPredict(int imageNumber);
    void showImageInLabel(int imageNumber);
signals:
    void sendData(QString data);
    //void sendImageResult(QString predResult);
public slots:
    void clickBtnTrain();
    void clickBtnVal();
    void setTestImageNum();
    void updateTextEditData(QString data);
    void predImageClass();
private:
    Ui::ConvNeuralNetworkWidgetClass ui;
    int testImageNumber = 0;
};
