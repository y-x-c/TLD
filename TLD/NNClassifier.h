//
//  NNClassifier.h
//  TLD
//
//  Created by 陈裕昕 on 11/4/14.
//  Copyright (c) 2014 Fudan. All rights reserved.
//

#ifndef __TLD__NNClassifier__
#define __TLD__NNClassifier__

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

class NNClassifier
{
private:
    static const int thModelSize = 100;
    static const int patchSize = 15;
    constexpr static const float thMargin = 0.1;
    
    float thNN = 0.6;
    
    //save positive patches and negative patches in [1 x (patchSize ^ 2)]
    vector<Mat> pPatches, nPatches;

    float calcNCC(const Mat &patch1, const Mat &patch2);
    
    Mat getPatch(const Mat &img);
    
public:
    const bool cPos = true; // positive
    const bool cNeg = false;
    
    typedef pair<Mat, bool> tTrainData;
    typedef vector<tTrainData> tTrainDataSet;
    
    NNClassifier();
    
    ~NNClassifier();
    
    void update(const Mat &patch, int c);
    void trainInit(const tTrainDataSet &trainDataSet);
    void train(const tTrainDataSet &trainDataSet);
    bool getClass(const Mat &img);
    
    void showModel();
    
    float calcSP(const Mat &img);
    float calcSPHalf(const Mat &img);
    float calcSN(const Mat &img);
    float calcSr(const Mat &img);
    float calcSc(const Mat &img);
};

#endif /* defined(__TLD__NNClassifier__) */
