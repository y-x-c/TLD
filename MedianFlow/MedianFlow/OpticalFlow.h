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

#include "TLDSystemStruct.h"

using namespace std;
using namespace cv;

class OpticalFlow
{
private:
    Mat prevImg, nextImg;
    
    ///// my own implementaion of optical flow
    
    // Size of the window centered by the traced point.
    // The value must be odd.
    const static int neighborSize = 5;
    const static int maxLevel = 0; // 0 means no pyramid
    
    vector<Mat> prevImgs, nextImgs, Ixs, Iys, Its;
    
    bool isInside(const Point2f &pt, int imgWidth, int imgHeight);
    
    void getIxy(const Mat &img, Mat &ret, int dx, int dy);
    
    void preprocess();
    Point2f calculate(const Point2f &trackPoint, const Mat &Ix, const Mat &Iy, const Mat &It);
    Point2f calculatePyr(const Point2f &trackPoint);

    vector<Point2f> generateNeighborPts(const Point2f &pt, int imgWidth, int imgHeight);
    /////
    
public:
    OpticalFlow();
    
    // prevImg & nextImg should be CV_8U | CV_8UC3
    OpticalFlow(const Mat &prevImg, const Mat &nextImg);
    
    ~OpticalFlow();
    

    void trackPts(vector<TYPE_OF_PT> &pts, vector<TYPE_OF_PT> &retPts, vector<uchar> &status);

    // To make MedianFlow can be used repeatedly fast,
    // creating two OF instances is much better.
    // So this function was removed.
    //void swapImg();
};

#endif /* defined(__MedianFlow__OpticalFlow__) */
