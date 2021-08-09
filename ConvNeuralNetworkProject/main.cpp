#include "ConvNeuralNetworkWidget.h"
#include <QtWidgets/QApplication>
#include <armadillo>

using namespace std;
using namespace arma;

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    ConvNeuralNetworkWidget w;
    w.show();
    return a.exec();
}
