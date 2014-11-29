//
//  OpticalFlow.cpp
//  MedianFlow
//
//  Created by 陈裕昕 on 10/15/14.
//  Copyright (c) 2014 陈裕昕. All rights reserved.
//

#include "OpticalFlow.h"

OpticalFlow::OpticalFlow()
{

}

OpticalFlow::OpticalFlow(const Mat &prevImg, const Mat &nextImg)
{
    this->prevImg = prevImg;
    this->nextImg = nextImg;
}

OpticalFlow::~OpticalFlow()
{

}

void OpticalFlow::trackPts(vector<TYPE_OF_PT> &pts, vector<TYPE_OF_PT> &retPts, vector<uchar> &status)
{
    vector<float> err;
        
    calcOpticalFlowPyrLK(prevImg, nextImg, pts, retPts, status, err, Size(21, 21), 5, cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 30, 0.01), OPTFLOW_USE_INITIAL_FLOW);
}