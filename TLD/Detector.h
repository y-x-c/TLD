//
//  Detector.h
//  TLD
//
//  Created by 陈裕昕 on 11/5/14.
//  Copyright (c) 2014 Fudan. All rights reserved.
//

#ifndef __TLD__Detector__
#define __TLD__Detector__

#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/legacy.hpp>

#include "VarClassifier.h"
#include "RandomFernsClassifier.h"
#include "NNClassifier.h"
#include "Learner.h"

#include "TLDSystemStruct.h"

using namespace std;
using namespace cv;

class Detector
{
    friend class Learner;
private:
    RandomFernsClassifier rFClassifier;
    NNClassifier nNClassifier;
    
    Mat pattern;
    TYPE_DETECTOR_BB patternBB;
    int imgW, imgH;
    float patternVar;
    PatchGenerator patchGenerator;
    PatchGenerator updatePatchGenerator;
    TYPE_TRAIN_DATA_SET trainDataSetNN, trainDataSetRF;
    
    TYPE_DETECTOR_RET rfRet;
    
    vector<float> scales;
    
    float overlap(const TYPE_DETECTOR_BB &bb1, const TYPE_DETECTOR_BB &bb2);
    
    // scanning-window grid
    TYPE_DETECTOR_SCANBBS scanBBs;
    void genScanBB();
    
    void sortByOverlap(const TYPE_DETECTOR_BB &bb, bool rand = false);
    
    //void genWarped(const Mat &img, Mat &warped);
    //void genUpdateWarped(const Mat &img, Mat &warped);
    void genPosData(const Mat &img, const Mat &imgB, const Mat &img32F, TYPE_TRAIN_DATA_SET &trainDataSetNN, TYPE_TRAIN_DATA_SET &trainDataSetRF, const int nWarped = DETECTOR_N_WARPED);
    void genNegData(const Mat &img, const Mat &imgB, const Mat &img32F, TYPE_TRAIN_DATA_SET &trainDataSetNN, TYPE_TRAIN_DATA_SET &trainDataSetRF);
    
    void update();
    void train(const Mat &img, const Mat &imgB, const Mat &img32F, const Rect &patternBB);
    
public:
    Detector(){}
    void init(const Mat &img, const Mat &imgB, const Mat &img32F, const Rect &patternBB);
    
    void dectect(const Mat &img, const Mat &imgB, const Mat &img32F, TYPE_DETECTOR_RET &ret);
    
    void updataNNPara(const Mat &img32F, TYPE_DETECTOR_SCANBB &sbb);
    float getNNThPos();
    
    ~Detector();
};

#endif /* defined(__TLD__Detector__) */
