//
//  Learner.cpp
//  TLD
//
//  Created by 陈裕昕 on 14/11/7.
//  Copyright (c) 2014年 Fudan. All rights reserved.
//

#include "Learner.h"

Learner::Learner(Detector *_detector):
    detector(*_detector)
{
    
}

Learner::~Learner()
{
    
}

void Learner::learn(const Mat &img, const Rect &ret)
{
    cerr << "Start learning" << endl;
    
    cerr << "Sort bounding boxes." << endl;
    detector.sortByOverlap(ret);
    detector.trainDataSet.clear();
    
    //P-expert
    cerr << "Run P-expert" << endl;
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < nWarped; j++)
        {
            Mat warped;
            detector.genWarped(img(detector.scanBBs[i].first), warped);
            
            Detector::tTrainData trainData(make_pair(warped, detector.cPos));
            detector.trainDataSet.push_back(trainData);
        }
    }
    
    //N-expert
    cerr << "Run N-expert" << endl;
    for(int i = nWarped; i < detector.scanBBs.size(); i++)
    {
        if(detector.scanBBs[i].status == Detector::bbRejNN)
        {
            Detector::tTrainData trainData(make_pair(img(detector.scanBBs[i].first), detector.cNeg));
            detector.trainDataSet.push_back(trainData);
        }
    }
    
    detector.update();
    
    cerr << "Finish learning" << endl;
}