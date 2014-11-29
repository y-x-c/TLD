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
#include <map>

using namespace std;
using namespace cv;

class LeafCounter
{
    int p, n;
    float posteriors;
    
public:
    LeafCounter()
    {
        p = n = 0;
        posteriors = 0.f;
    }
    
    void voteP()
    {
        p++;
        posteriors = (float)p / (p + n);
    }
    
    void voteN()
    {
        n++;
        posteriors = (float)p / (p + n);
    }
    
    float getPosteriors()
    {
        return posteriors;
    }
};

typedef vector<LeafCounter> FernCounter;

class CmpPt
{
    int p1x, p1y, p2x, p2y;
    
public:
    CmpPt(int _p1x, int _p1y, int _p2x, int _p2y)
    {
        p1x = _p1x; p1y = _p1y; p2x = _p2x; p2y = _p2y;
    }
    
    void get(int &_p1x, int &_p1y, int &_p2x, int &_p2y)
    {
        _p1x = p1x; _p1y = p1y; _p2x = p2x; _p2y = p2y;
    }
};

class RandomFernsClassifier
{
private:
    float thPos;
    
    int nFerns, nLeaves;
    vector<FernCounter> counter;
    vector<vector<float> > posteriors;
    TYPE_FERN_FERNS ferns;
    
    vector<vector<vector<CmpPt> > > cmpPts; // [scaleId][fernId][leafId]
    map<pair<int, int>, int> scalesId;
    
    void update(const Mat &img, bool c, float p = -1.);
    
    float getRNG();
    int getCode(const Mat &img, int idx);
    
    float getPosteriors(const Mat &img);
    float getSumPosteriors(const Mat &img);
    
    void gen4Pts(float ox, float oy, vector<TYPE_FERN_LEAF> &tleave);
public:
    RandomFernsClassifier();
    void init(int nStructs, int nFerns, const vector<float> &scales, int initW, int initH);
    
    ~RandomFernsClassifier();
    bool getClass(const Mat &img, TYPE_DETECTOR_SCANBB &sbb);
    
    void train(const TYPE_TRAIN_DATA_SET &trainDataSet);
};

#endif /* defined(__TLD__RandomFernsClassifier__) */
