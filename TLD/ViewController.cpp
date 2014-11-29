//
//  ViewController.cpp
//  MedianFlow
//
//  Created by 陈裕昕 on 10/16/14.
//  Copyright (c) 2014 陈裕昕. All rights reserved.
//

#include "ViewController.h"


ViewController::~ViewController()
{
    
}

ViewController::ViewController(VideoController *videoController):
    retWindowName("TLD")
{
    this->videoController = videoController;
}

void ViewController::refreshCache()
{
    if(videoController->cameraMode)
        videoController->readNextFrame();
    videoController->getCurrFrame().copyTo(cache);
}

void ViewController::drawLines(const vector<Point2f> &firstPts, const vector<Point2f> &secondPts, const Scalar &color, int thickness)
{
    for(int i = 0; i < firstPts.size(); i++)
    {
        if(secondPts[i] == PT_ERROR) continue;
        line(cache, firstPts[i], secondPts[i], color, thickness);
    }
}

void ViewController::drawCircles(const vector<Point2f> &pts, const Scalar &color, int radius)
{
    for(int i = 0; i < pts.size(); i++)
    {
        circle(cache, pts[i], radius, color);
    }
}

void ViewController::showCache(const string &winName)
{
    if(winName.size())
        imshow(winName, cache);
    else
        imshow(retWindowName, cache);
    
    waitKey(1);
}

void ViewController::drawRect(const Rect_<float> &rect, const Scalar &color, int thickness)
{
    rectangle(cache, rect, color, thickness);
}

void ViewController::onMouse(int event, int x, int y, int flags, void* param)
{
    pair<pair<void*, void*>, bool*> pp = *(pair<pair<void*, void*>, bool*>*)(param);
    pair<void*, void*> p = pp.first;
    
    bool &selectValid = *((bool*)pp.second);
    Rect &rect = *((Rect*)p.first);
    ViewController &viewController = *((ViewController*)p.second);
    int width = viewController.videoController->getCurrFrame().cols;
    int height = viewController.videoController->getCurrFrame().rows;
    
    static int x0 = -1, y0 = -1;
    
    if(event == CV_EVENT_LBUTTONDOWN)
    {
        selectValid = false;
        x0 = x; y0 = y;
        rect = Rect(Point2i(x0, y0), Point2i(x0, y0));
    }
    
    if(flags == CV_EVENT_FLAG_LBUTTON && event != CV_EVENT_LBUTTONDOWN)
    {
        rect = Rect(Point2i(min(x0, x), min(y0, y)), Point2i(max(x0, x), max(y0, y)));
        
        viewController.refreshCache();
        viewController.drawRect(rect);
        viewController.showCache(viewController.retWindowName);
        
        if(rect.width >= DETECTOR_MIN_BB_SIZE  && rect.height >= DETECTOR_MIN_BB_SIZE && rect.width <= width && rect.height <= height)
        {
            selectValid = true;
        }
    }
    
    if(event == CV_EVENT_LBUTTONUP)
    {
        if(rect.width >= DETECTOR_MIN_BB_SIZE && rect.height >= DETECTOR_MIN_BB_SIZE && rect.width <= width && rect.height <= height)
        {
            selectValid = true;
        }
        else
        {
            rect = BB_ERROR;
        }
    }
}

Rect ViewController::getRect()
{
    namedWindow(retWindowName, CV_WINDOW_AUTOSIZE);
    
    imshow(retWindowName, videoController->getCurrFrame());
    
    
    int width = videoController->getCurrFrame().cols;
    int height = videoController->getCurrFrame().rows;
    
    Rect rect(Point2d(), Point2d(width + 1, height + 1));
    pair<void*, void*> p(&rect, this);
    
    bool selectValid = false;
    
    pair<pair<void*, void*>, bool*> pp(p, &selectValid);
    
    setMouseCallback(retWindowName, ViewController::onMouse, &pp);
    
    if(videoController->cameraMode)
    {
        while(true)
        {
            refreshCache();
            drawRect(rect);
            imshow(retWindowName, cache);
            int f = waitKey(1);
            if(selectValid && (f != -1)) break;
        }
    }
    else
    {
        while(!selectValid)
        {
            waitKey();
        }
    }
    
    destroyWindow(retWindowName);
    
    return rect;
}