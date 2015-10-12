//
//  MedianFlow.h
//  MedianFlow
//
//  Created by 陈裕昕 on 10/29/14.
//  Copyright (c) 2014 陈裕昕. All rights reserved.
//

#ifndef __MedianFlow__MedianFlow__
#define __MedianFlow__MedianFlow__


#include "../../TLD/TLDSystemStruct.h"

#include <cmath>
#include <iostream>

#include "OpticalFlow.h"

// show result
#include "../../TLD/ViewController.h"
//

using namespace std;
using namespace cv;

class MedianFlow
{
private:
    Mat prevImg, nextImg;
    
    OpticalFlow *opticalFlow, *opticalFlowSwap;
    ViewController *viewController;
    
    bool isPointInside(const TYPE_MF_PT &pt, const TYPE_MF_COORD border = 0);
    bool isBoxUsable(const TYPE_MF_BB &rect);
    
    void generatePts(const TYPE_MF_BB &box, vector<TYPE_MF_PT> &ret);
    
    float calcNCC(const Mat &img0, const Mat &img1);
    
    void filterOFError(const vector<TYPE_MF_PT> &pts, const vector<uchar> &retF, vector<int> &rejected);
    void filterFB(const vector<TYPE_MF_PT> &initialPts, const vector<TYPE_MF_PT> &FBPts, vector<int> &rejected);
    void filterNCC(const vector<TYPE_MF_PT> &initialPts, const vector<TYPE_MF_PT> &FPts, vector<int> &rejected);
    
    TYPE_MF_BB calcRect(const TYPE_MF_BB &rect, const vector<TYPE_MF_PT> &pts, const vector<TYPE_MF_PT> &FPts,  const vector<TYPE_MF_PT> &FBPts, const vector<int> &rejected, int &status);
    
public:
    
    MedianFlow();
 
    // prevImg & nextImg should be CV_8U
    // viewController for showing MedianFlow results
    MedianFlow(const Mat &prevImg, const Mat &nextImg, ViewController *viewController = NULL);
    
    ~MedianFlow();
    
    static bool compare(const pair<float, int> &a, const pair<float, int> &b);
    
    TYPE_MF_BB trackBox(const TYPE_MF_BB &inputBox, int &status);
};

#endif /* defined(__MedianFlow__MedianFlow__) */
