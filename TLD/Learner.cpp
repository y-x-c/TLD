//
//  Learner.cpp
//  TLD
//
//  Created by 陈裕昕 on 14/11/7.
//  Copyright (c) 2014年 Fudan. All rights reserved.
//

#include "Learner.h"

Learner::Learner(Detector *_detector):
    detector(_detector)
{
    
}

Learner::~Learner()
{
    
}

void Learner::learn(const Mat &img, const Rect &ret)
{
    cerr << "Start learning" << endl;
 
    int imgW = img.cols, imgH = img.rows;
    Rect rImg(0, 0, imgW, imgH);
    if(!(rImg.contains(ret.tl()) && rImg.contains(ret.br())))
    {
        cerr << "Learning exited because bb is out of image." << endl;
        return;
    }
    
    detector->sortByOverlap(ret, false);
    detector->trainDataSet.clear();
    
    //P-expert
    int pCount = 0;
    for(int i = 0; i < LEARNER_N_GOOD_BB; i++)
    {
        if(detector->calcSr(img(detector->scanBBs[i].first)) < LEARNER_TH_GOODBB_SR) continue;
        
        for(int j = 0; j < LEARNER_N_WARPED; j++)
        {
            pCount++;
            
            Mat warped;
            detector->genWarped(img(detector->scanBBs[i].first), warped);
            
            TYPE_TRAIN_DATA trainData(make_pair(warped, CLASS_POS));
            detector->trainDataSet.push_back(trainData);
        }
    }
    
    //N-expert
    int nCount = 0;
    for(int i = LEARNER_N_GOOD_BB; i < detector->scanBBs.size(); i++)
    {
        if(detector->scanBBs[i].status == DETECTOR_REJECT_NN)
        {
            nCount++;
            
            TYPE_TRAIN_DATA trainData(make_pair(img(detector->scanBBs[i].first), CLASS_NEG));
            detector->trainDataSet.push_back(trainData);
        }
    }
    
    cerr << "Generate " << pCount << " positive sample(s) and " << nCount << " negative sample(s)." << endl;
    
    cerr << "Update detector" << endl;
    detector->update();
    
    cerr << "Finish learning" << endl;
    
    detector->nNClassifier.showModel();
}