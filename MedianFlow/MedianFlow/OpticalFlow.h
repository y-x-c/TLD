//
//  OpticalFlow.h
//  MedianFlow
//
//  Created by 陈裕昕 on 10/15/14.
//  Copyright (c) 2014 陈裕昕. All rights reserved.
//

#ifndef __MedianFlow__OpticalFlow__
#define __MedianFlow__OpticalFlow__

#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "../../TLD/TLDSystemStruct.h"

using namespace std;
using namespace cv;

class OpticalFlow
{
private:
    Mat prevImg, nextImg;
    
public:
    OpticalFlow();
    
    // prevImg & nextImg should be CV_8U
    OpticalFlow(const Mat &prevImg, const Mat &nextImg);
    
    ~OpticalFlow();
    
    void trackPts(vector<TYPE_OF_PT> &pts, vector<TYPE_OF_PT> &retPts, vector<uchar> &status);

};

#endif /* defined(__MedianFlow__OpticalFlow__) */
