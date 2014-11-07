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
#include <opencv2/opencv.hpp>

#include "MedianFlow.h"
#include "Detector.h"
#include "Learner.h"

using namespace std;
using namespace cv;

class TLD
{
private:
    MedianFlow *tracker;
    Detector detector;
    Learner learner;
    
    Mat prevImg, nextImg;
    Rect bb;
    
public:
    
    TLD(const Mat &img, const Rect &bb);
    
    ~TLD();
    
    void setNextFrame(const Mat &frame);
    void track();
    
    Rect getBB();
    
};

#endif /* defined(__TLD__TLD__) */
