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

using namespace std;
using namespace cv;

class RandomFernsClassifier
{
public:
    typedef pair<Point2f, Point2f> tLeaf;
    typedef vector<vector<tLeaf> > tFerns;
    typedef pair<Mat, bool> tTrainData;
    typedef vector<tTrainData> tTrainDataSet;
    typedef vector<pair<int, int> > tPNCounter;
    const bool cPos = true;
    const bool cNeg = false;
    
private:
    const float pTh = 0.5;
    const float nTh = 1 - pTh;
    
    int nFerns, nLeaves;
    vector<tPNCounter> counter;
    tFerns ferns;
    
public:
    
    RandomFernsClassifier();
    RandomFernsClassifier(int nStructs, int nFerns);
    
    ~RandomFernsClassifier();
    
    void update(const Mat &img, bool c, float p = -1.);
    
    int getCode(const Mat &img, int idx);
    float getPosteriors(const Mat &img);
    bool getClass(const Mat &img);
    
    void train(const tTrainDataSet &trainDataSet);
};

#endif /* defined(__TLD__RandomFernsClassifier__) */
