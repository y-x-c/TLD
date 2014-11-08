//
//  TLD.cpp
//  TLD
//
//  Created by 陈裕昕 on 14/11/7.
//  Copyright (c) 2014年 Fudan. All rights reserved.
//

#include "TLD.h"

TLD::TLD(const Mat &img, const Rect &_bb):
bb(_bb)
{
    setNextFrame(img);
    
    detector = Detector(nextImg, _bb);
    learner = Learner(&detector);
}

TLD::~TLD()
{
    
}

void TLD::setNextFrame(const cv::Mat &frame)
{
    cv::swap(prevImg, nextImg);
    
    cvtColor(frame, nextImg, CV_BGR2GRAY);
    GaussianBlur(nextImg, nextImg, Size(9, 9), 1.5);
}

void TLD::track(Rect &bbTrack, vector<Rect> &bbDetect)
{
    ///// debug
    bbTrack = BB_ERROR;
    bbDetect.clear();
    /////
    
    tracker = new MedianFlow(prevImg, nextImg);
    
    //track
    int trackerStatus;
    Rect trackerRet = tracker->trackBox(bb, trackerStatus);
    
    //detect
    TYPE_DETECTOR_RET detectorRet;
    detector.dectect(nextImg, detectorRet);
    
    //integrate
    float maxSc = -1;
    Rect maxBB;
    
    
    ////// just test
    int tlx = max(0, trackerRet.tl().x), tly = max(0, trackerRet.tl().y);
    int brx = min(nextImg.cols, trackerRet.br().x), bry = min(nextImg.rows, trackerRet.br().y);
    Rect _rect(tlx, tly, brx - tlx, bry - tly);
    
    if(trackerStatus == MF_TRACK_SUCCESS && detector.calcSc(nextImg(_rect)) < 0.4)
    {
        trackerStatus = !trackerStatus;
    }
    //////
    
    if(trackerStatus != MF_TRACK_SUCCESS && detectorRet.size() == 0)
    {
        cerr << "Not visible." << endl;
        bb = BB_ERROR;
        
        delete tracker;
        return;
    }
    
    if(trackerStatus == MF_TRACK_SUCCESS)
    {
        int tlx = max(0, trackerRet.tl().x), tly = max(0, trackerRet.tl().y);
        int brx = min(nextImg.cols, trackerRet.br().x), bry = min(nextImg.rows, trackerRet.br().y);
        Rect _rect(tlx, tly, brx - tlx, bry - tly);
        
        float Sc = detector.calcSc(nextImg(_rect));
        cerr << "track bb Sc : " << Sc << " Sr : " << detector.calcSr(nextImg(_rect)) <<  endl;
        if(Sc > maxSc)
        {
            maxSc = Sc;
            maxBB = trackerRet;
        }
        
        bbTrack = trackerRet;
    }
    
    if(detectorRet.size())
    {
        for(auto &bb : detectorRet)
        {
            float Sc = detector.calcSc(nextImg(bb));
            cerr << "detect bb Sc : " << Sc << " Sr : " << detector.calcSr(nextImg(bb)) << endl;
            if(Sc > maxSc)
            {
                maxSc = Sc;
                maxBB = bb;
            }
            
            bbDetect.push_back(bb);
        }
    }
    
    if(detector.calcSr(nextImg(maxBB == bbTrack ? _rect : maxBB)) >= 0.5)
    {
        learner.learn(nextImg, maxBB);
    }

    bb = maxBB;

    delete tracker;
}

Rect TLD::getBB()
{
    return bb;
}