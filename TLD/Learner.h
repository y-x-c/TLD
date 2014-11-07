//
//  Learner.h
//  TLD
//
//  Created by 陈裕昕 on 14/11/7.
//  Copyright (c) 2014年 Fudan. All rights reserved.
//

#ifndef __TLD__Learner__
#define __TLD__Learner__

#include <iostream>
#include <opencv2/opencv.hpp>

#include "Detector.h"
class Detector;

using namespace std;
using namespace cv;

class Learner
{
public:
    static const int nWarped = 10;
    
private:
    Detector &detector;
    
public:
    
    Learner(Detector *detector);
    
    ~Learner();
    
    void learn(const Mat &img, const Rect &ret);
};

#endif /* defined(__TLD__Learner__) */
