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
#include "TLDSystemStruct.h"

using namespace std;
using namespace cv;

class NNClassifier
{
private:
    //save positive patches and negative patches in [1 x (patchSize ^ 2)]
    vector<Mat> pPatches, nPatches;

    float calcNCC(const Mat &patch1, const Mat &patch2);
    
    Mat getPatch(const Mat &img);
    
    bool update(const Mat &patch, int c);
    
    float thPos;
    
    Mat newSamplesP, newSamplesN;
    void addToNewSamples(const Mat &patch, const int c);
    
public:
    NNClassifier();
    
    ~NNClassifier();
    
    void trainInit(const TYPE_TRAIN_DATA_SET &trainDataSet);
    void train(const TYPE_TRAIN_DATA_SET &trainDataSet);
    
    void showModel();
    
    // assert : img.type() == CV_8U
    bool getClass(const Mat &img);
    
    float calcSP(const Mat &img);
    float calcSPHalf(const Mat &img);
    float calcSN(const Mat &img);
    float calcSr(const Mat &img);
    float calcSc(const Mat &img);
};

#endif /* defined(__TLD__NNClassifier__) */
