#include "MaxPoolLyaer.h"
#include "Util.h"

shared_ptr<DataStruct> MaxPoolLyaer::forward(DataStruct& A_prev)
{
    vector<shared_ptr<dcube>>& x = A_prev.getData();
    vector<int> xShape = A_prev.getSize().getSize();
    int m = xShape[0];
    int h = xShape[1];
    int w = xShape[2];
    int c = xShape[3];
    int feature_w = floor(w / 2);
    int feature_h = floor(h / 2);
    DataStruct feature(DataSize(m, feature_h, feature_w, c));
    feature.createZeroVecCubeBySize();
    vector<shared_ptr<dcube>>& f_tensor = feature.getData();
    //mask,记录最大池化时，最大值的位置信息;
    DataStruct featureMask(DataSize(m, h, w, c));
    featureMask.createZeroVecCubeBySize();
    vector<shared_ptr<dcube>>& fMask_tensor = featureMask.getData();

    for (int mi = 0; mi < m; mi++) {
        for (int ci = 0; ci < c; ci++) {
            for (int i = 0; i < feature_h; i++) {
                for (int j = 0; j < feature_w; j++) {
                    mat x_slice = (*(x[mi])).slice(ci);
                    mat x_k = x_slice(span(i * 2, i * 2 + 2-1), span(j * 2, j * 2 + 2-1));
                    mat maxx = max(max(x_k, 0), 1);
                    (*(f_tensor[mi]))(i, j, ci) = maxx(0, 0);
                    int index = x_slice(span(i * 2, i * 2 + 2-1), span(j * 2, j * 2 + 2-1)).index_max();
                    (*(fMask_tensor[mi]))(i*2+index%2,j*2+index/2,ci) = 1;
                }
            }
        }
    }
    // m, h, w, c
    this->pfeatureMask = make_shared<DataStruct>(featureMask);
    //m, feature_h, feature_w, c
    return make_shared<DataStruct>(feature);
}

shared_ptr<DataStruct> MaxPoolLyaer::backward(DataStruct& dZ)
{
    // m, feature_h, feature_w,c
    vector<int> dZSize =  dZ.getSize().getSize();
    int feature_h = dZSize[1];
    int feature_w = dZSize[2];
    int c = dZSize[3];
    vector<shared_ptr<dcube>> result;
    dcube maskedRepeat_dZ;
    for (int i = 0; i < dZ.getData().size(); i++) {
        dcube repeat_dZ(2 * feature_h, 2 * feature_w, c);
        Util::repeatRowCol2(*(dZ.getData()[i]), repeat_dZ);
        //点乘
        maskedRepeat_dZ = repeat_dZ % (*(this->pfeatureMask->getData()[i]));
        shared_ptr<dcube> p = make_shared<dcube>(maskedRepeat_dZ);
        result.push_back(p);
    }
    shared_ptr<DataStruct> pResult = make_shared<DataStruct>(result);

    return pResult;
}
