//
//  RandomFernsClassifier.h
//  TLD
//
//  Created by 陈裕昕 on 11/4/14.
//  Copyright (c) 2014 Fudan. All rights reserved.
//

#ifndef __TLD__RandomFernsClassifier__
#define __TLD__RandomFernsClassifier__

#include <iostream>
#include <opencv2/opencv.hpp>
#include "TLDSystemStruct.h"

using namespace std;
using namespace cv;

class RandomFernsClassifier
{
private:
    float thPos;
    
    int nFerns, nLeaves;
    vector<TYPE_FERN_PNCOUNTER> counter;
    TYPE_FERN_FERNS ferns;
    
    void update(const Mat &img, bool c, float p = -1.);
    
    float getRNG();
    int getCode(const Mat &img, int idx);
    
    float getPosteriors(const Mat &img);
    
public:
    
    RandomFernsClassifier();
    RandomFernsClassifier(int nStructs, int nFerns);
    
    ~RandomFernsClassifier();
    bool getClass(const Mat &img, TYPE_DETECTOR_SCANBB &sbb);
    
    void train(const TYPE_TRAIN_DATA_SET &trainDataSet);
};

#endif /* defined(__TLD__RandomFernsClassifier__) */
