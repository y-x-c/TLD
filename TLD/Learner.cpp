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

void Learner::learn(const Mat &img, const Mat &imgB, const Rect &ret)
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
    
    TYPE_TRAIN_DATA_SET &nnTrainDataset = detector->trainDataSetNN;
    nnTrainDataset.clear();
    TYPE_TRAIN_DATA_SET &rfTrainDataset = detector->trainDataSetRF;
    rfTrainDataset.clear();
    
    detector->genPosData(img, imgB, nnTrainDataset, rfTrainDataset);
    
    //P-expert
    int pCount = 1;
//    nnTrainDataset.push_back(make_pair(img(ret), CLASS_POS));
    
    //N-expert
    int nCount = 0;
    for(int i = 0; i < detector->scanBBs.size(); i++)
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

    
    pCount = (int)rfTrainDataset.size();
    
    //N-expert
    nCount = 0;
    for(int i = 0; i < detector->scanBBs.size(); i++)
    {
        TYPE_DETECTOR_SCANBB &sbb = detector->scanBBs[i];
        
        if(sbb.status != DETECTOR_REJECT_VAR)
        {
            if(sbb.second < LEARNER_TH_OL && detector->rFClassifier.getPosteriors(imgB(sbb.first)) >= 0.1)
            {
                nCount++;
                
                TYPE_TRAIN_DATA trainData(make_pair(imgB(sbb.first), CLASS_NEG));
                rfTrainDataset.push_back(trainData);
            }
        }
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