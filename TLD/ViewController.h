//
//  ViewController.h
//  MedianFlow
//
//  Created by 陈裕昕 on 10/16/14.
//  Copyright (c) 2014 陈裕昕. All rights reserved.
//

#ifndef __MedianFlow__ViewController__
#define __MedianFlow__ViewController__

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "VideoController.h"

using namespace std;
using namespace cv;

class ViewController
{
private:
    VideoController *videoController;
    
    Mat cache;
    
    string retWindowName;
    
public:
    
    ViewController();
    
    ViewController(VideoController *videoController);
    
    ~ViewController();
    
    void drawCircles(const vector<Point2f> &pts, Scalar color = Scalar(255, 255, 255), int radius = 3);
    
    void drawLines(const vector<Point2f> &firstPts, const vector<Point2f> &secondPts, Scalar color = Scalar(78, 86, 255));
    
    void showCache(const string &winName = string());
    
    void drawRect(const Rect_<float> &rect);
    
    Rect_<float> getRect();
    
    static void onMouse(int event, int x, int y, int, void *);
    
    void refreshCache();
};

#endif /* defined(__MedianFlow__ViewController__) */
