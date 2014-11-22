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
    
    //////////////////////////
    // NN's training dataset//
    //////////////////////////
    
    TYPE_TRAIN_DATA_SET nnTrainDataset;
    
    //P-expert
    int pCount = 0;
    for(int i = 0; i < LEARNER_N_GOOD_BB; i++)
    {
        TYPE_DETECTOR_SCANBB &sbb = detector->scanBBs[i];
        
        if(sbb.status == DETECTOR_REJECT_NN)
        {
            for(int j = 0; j < LEARNER_N_WARPED; j++)
            {
                pCount++;
                
                Mat warped;
                detector->genUpdateWarped(img(sbb.first), warped);
                
                TYPE_TRAIN_DATA trainData(make_pair(warped, CLASS_POS));
                nnTrainDataset.push_back(trainData);
            }
        }
    }
    
    //N-expert
    int nCount = 0;
    for(int i = LEARNER_N_GOOD_BB; i < detector->scanBBs.size(); i++)
    {
        TYPE_DETECTOR_SCANBB &sbb = detector->scanBBs[i];
        
        if(sbb.status == DETECTOR_ACCEPTED)
        {
            if(sbb.second < LEARNER_TH_OL)
            {
                nCount++;
                
                TYPE_TRAIN_DATA trainData(make_pair(img(sbb.first), CLASS_NEG));
                nnTrainDataset.push_back(trainData);
            }
        }
    }
    
    cerr << "Generate " << pCount << " positive sample(s) and " << nCount << " negative sample(s) for NN classifier" << endl;
    
    // end NN's training dataset
    
    //////////////////////////
    // RF's training dataset//
    //////////////////////////
    
    TYPE_TRAIN_DATA_SET rfTrainDataset;

    //P-expert
    pCount = 0;
    for(int i = 0; i < LEARNER_N_GOOD_BB; i++)
    {
        TYPE_DETECTOR_SCANBB &sbb = detector->scanBBs[i];
        
        if(true)
        {
            for(int j = 0; j < LEARNER_N_WARPED; j++)
            {
                pCount++;
                
                Mat warped;
                detector->genUpdateWarped(img(sbb.first), warped);
                
                TYPE_TRAIN_DATA trainData(make_pair(warped, CLASS_POS));
                rfTrainDataset.push_back(trainData);
            }
        }
    }
    
    //N-expert
    nCount = 0;
    for(int i = LEARNER_N_GOOD_BB; i < detector->scanBBs.size(); i++)
    {
        TYPE_DETECTOR_SCANBB &sbb = detector->scanBBs[i];
        
        if(sbb.status != DETECTOR_REJECT_VAR || sbb.status != DETECTOR_REJECT_RF)
        {
            if(sbb.second < LEARNER_TH_OL)
            {
                nCount++;
                
                TYPE_TRAIN_DATA trainData(make_pair(img(sbb.first), CLASS_NEG));
                rfTrainDataset.push_back(trainData);
            }
        }
        
        ///debug
        if(nCount == 5000) break;
        ///end debug
    }
    
    cerr << "Generate " << pCount << " positive sample(s) and " << nCount << " negative sample(s) for RF classifier" << endl;
    // end RF's training dataset
    
    cerr << "Update detector" << endl;
    //detector->update();
    detector->rFClassifier.train(rfTrainDataset);
    detector->nNClassifier.train(nnTrainDataset);
    
    cerr << "Finish learning" << endl;
    
    detector->nNClassifier.showModel();
}