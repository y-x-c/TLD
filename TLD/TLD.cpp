//
//  TLD.cpp
//  TLD
//
//  Created by 陈裕昕 on 14/11/7.
//  Copyright (c) 2014年 Fudan. All rights reserved.
//

#include "TLD.h"

TLD::TLD(const Mat &img, const Rect &_bb):
detector(img, _bb), learner(&detector), bb(_bb)
{
    cvtColor(img, nextImg, CV_BGR2GRAY);
}

TLD::~TLD()
{
    
}

void TLD::setNextFrame(const cv::Mat &frame)
{
    cv::swap(prevImg, nextImg);
    
    cvtColor(frame, nextImg, CV_BGR2GRAY);
}

void TLD::track()
{
    tracker = new MedianFlow(prevImg, nextImg);
    
    int trackerStatus;
    Rect trackBB = tracker->trackBox(bb, trackerStatus);
    
    Detector::tRet ret;
    
    detector.dectect(nextImg, ret);
    
    // integrator
    float maxSc = 0;
    Rect maxBB;
    
    for(auto &bb : ret)
    {
        float Sc = detector.calcSc(nextImg(bb));
        if(Sc > maxSc)
        {
            maxSc = Sc;
            maxBB = bb;
        }
    }
    
    if(trackerStatus == MedianFlow::MEDIANFLOW_TRACK_SUCCESS)
    {
        Point2d tl(max(0, trackBB.tl().x), max(0, trackBB.tl().y));
        Point2d br(min(prevImg.cols, trackBB.br().x), min(prevImg.rows, trackBB.br().y));
    
        float Sc = detector.calcSc(nextImg(Rect(tl, br)));
        if(Sc > maxSc)
        {
            maxSc = Sc;
            maxBB = trackBB;
            cerr << "Choose trace result" << endl;
        }
        else
        {
            cerr << "Choose detection result" << endl;
        }
    }
    else
    {
        if(ret.size() == 0)
            cerr << "Not visible." << endl;
        else
            cerr << "Choose detection result" << endl;
    }
    
    learner.learn(nextImg, maxBB);
    bb = maxBB;
    
    delete tracker;
}

Rect TLD::getBB()
{
    return bb;
}