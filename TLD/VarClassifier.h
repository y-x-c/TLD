//
//  VarClassifier.h
//  TLD
//
//  Created by 陈裕昕 on 14/11/6.
//  Copyright (c) 2014年 Fudan. All rights reserved.
//

#ifndef __TLD__VarClassifier__
#define __TLD__VarClassifier__

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class VarClassifier
{
public:
    constexpr static const float varFactor = 0.5;
    static const bool cPos = true;
    static const bool cNeg = false;
    
private:
    Mat sum, sqsum;
    
public:
    VarClassifier(){}
    VarClassifier(const Mat &img);
    
    ~VarClassifier(){}
    
    float getVar(const Rect &bb);
    bool getClass(const Rect &bb, float patternVar);
};

#endif /* defined(__TLD__VarClassifier__) */
