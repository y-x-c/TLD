//
//  TLD.h
//  TLD
//
//  Created by 陈裕昕 on 14/11/7.
//  Copyright (c) 2014年 Fudan. All rights reserved.
//

#ifndef __TLD__TLD__
#define __TLD__TLD__

#include <iostream>
#include <queue>
#include <opencv2/opencv.hpp>
#include <algorithm>

#include "MedianFlow.h"
#include "Detector.h"
#include "Learner.h"

#include "TLDSystemStruct.h"

using namespace std;
using namespace cv;

class TLD
{
private:
    //debug
public:
    // end debug
    MedianFlow *tracker;
    Detector detector;
    Learner learner;
    
    Mat prevImg, nextImg, nextImgB;
    Rect bb;
    
    bool valid;
    
    float overlap(const TYPE_DETECTOR_BB &bb1, const TYPE_DETECTOR_BB &bb2);
    Rect getInside(const Rect &bb);
    
public:
    
    TLD(const Mat &img, const Rect &bb);
    
    ~TLD();
    
    void setNextFrame(const Mat &frame);
    void track(Rect &bbTrack, vector<Rect> &bbDetect);
    
    Rect getBB();
    
};

#endif /* defined(__TLD__TLD__) */
