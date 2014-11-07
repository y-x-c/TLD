//
//  OpticalFlow.h
//  MedianFlow
//
//  Created by 陈裕昕 on 10/15/14.
//  Copyright (c) 2014 陈裕昕. All rights reserved.
//

#ifndef __MedianFlow__OpticalFlow__
#define __MedianFlow__OpticalFlow__

#define OFError Point2f(-1, -1)

#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class OpticalFlow
{
private:
    // Size of the window centered by the traced point.
    // The value must be odd.
    const static int neighborSize = 5;
    const static int maxLevel = 0; // 0 means no pyramid
    
    bool method;
    
    Mat prevImg, nextImg;
    vector<Mat> prevImgs, nextImgs, Ixs, Iys, Its;
    
    bool isInside(const Point2f &pt, int imgWidth, int imgHeight);
    
    void getIxy(const Mat &img, Mat &ret, int dx, int dy);
    
    void preprocess();
    Point2f calculate(const Point2f &trackPoint, const Mat &Ix, const Mat &Iy, const Mat &It);
    Point2f calculatePyr(const Point2f &trackPoint);

    vector<Point2f> generateNeighborPts(const Point2f &pt, int imgWidth, int imgHeight);
    
public:
    const static bool USEOPENCV = true;
    
    OpticalFlow();
    
    OpticalFlow(const Mat &prevImg, const Mat &nextImg, bool _method = USEOPENCV);
    
    ~OpticalFlow();
    
    void trackPts(vector<Point2f> &pts, vector<Point2f> &retPts);
    
    void swapImg();
};

#endif /* defined(__MedianFlow__OpticalFlow__) */
