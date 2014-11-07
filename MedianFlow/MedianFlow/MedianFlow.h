//
//  MedianFlow.h
//  MedianFlow
//
//  Created by 陈裕昕 on 10/29/14.
//  Copyright (c) 2014 陈裕昕. All rights reserved.
//

#ifndef __MedianFlow__MedianFlow__
#define __MedianFlow__MedianFlow__

//debug
#include "ViewController.h"
// end debug
#include "OpticalFlow.h"
#include <iostream>

using namespace std;
using namespace cv;

class MedianFlow
{
private:
    static const int halfPatchSize = 4;
    static const int nPts = 10;
    static const int errorDist = 20;
    
    Mat prevImg, nextImg;
    
    OpticalFlow *opticalFlow, *opticalFlowSwap;
    ViewController *viewController;
    
    bool isPointInside(const Point2f &pt, const float alpha = 0);
    bool isBoxUsable(const Rect_<float> &rect);
    
    vector<Point2f> generatePts(const Rect_<float> &box);
    
    double calcNCC(const Mat &img0, const Mat &img1);
    
    void filterOFError(const vector<Point2f> &pts, vector<int> &rejected);
    void filterFB(const vector<Point2f> &initialPts, const vector<Point2f> &FBPts, vector<int> &rejected);
    void filterNCC(const vector<Point2f> &initialPts, const vector<Point2f> &FPts, vector<int> &rejected);
    
    Rect_<float> calcRect(const Rect_<float> &rect, const vector<Point2f> &pts, const vector<Point2f> &FPts, const vector<int> &rejected, int &status);
    
public:
    
    MedianFlow();
    MedianFlow(const Mat &prevImg, const Mat &nextImg);
    MedianFlow(const Mat &prevImg, const Mat &nextImg, ViewController *_viewController);
    
    ~MedianFlow();
    
    static bool compare(const pair<float, int> &a, const pair<float, int> &b);
    
    static const int MEDIANFLOW_TRACK_SUCCESS = 0;
    static const int MEDIANFLOW_TRACK_F_PTS = -1; // number of points after filtering is too little
    static const int MEDIANFLOW_TRACK_F_BOX = -2; // box is out of bounds
    static const int MEDIANFLOW_TRACK_F_CONFUSION = -3; // tracking result is disordered
    static const int REJECT_OFERROR = 1 << 0;
    static const int REJECT_NCC = 1 << 1;
    static const int REJECT_FB = 1 << 2;
    Rect_<float> trackBox(const Rect_<float> &inputBox, int &status);
};

#endif /* defined(__MedianFlow__MedianFlow__) */
