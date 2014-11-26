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
    cerr << "[Learning]" << endl;
//    int imgW = img.cols, imgH = img.rows;
//    Rect rImg(0, 0, imgW, imgH);
//    if(!(rImg.contains(ret.tl()) && rImg.contains(ret.br())))
//    {
//        cerr << "Learning exited because bb is out of image." << endl;
//        return;
//    }
    
    detector->sortByOverlap(ret, false);
    
    TYPE_TRAIN_DATA_SET &nnTrainDataset = detector->trainDataSetNN;
    nnTrainDataset.clear();
    TYPE_TRAIN_DATA_SET &rfTrainDataset = detector->trainDataSetRF;
    rfTrainDataset.clear();
    
    // P-expert
    detector->genPosData(img, imgB, nnTrainDataset, rfTrainDataset, 10);
    
    // N-expert - NN
    int nCountNN = 0;
    for(int i = 0; i < detector->scanBBs.size(); i++)
    {
        TYPE_DETECTOR_SCANBB &sbb = detector->scanBBs[i];
        
        if(sbb.status != DETECTOR_REJECT_VAR && sbb.status != DETECTOR_REJECT_RF)
        {
            if(sbb.second < LEARNER_TH_OL)
            {
                nCountNN++;
                
                TYPE_TRAIN_DATA trainData(make_pair(img(sbb.first), CLASS_NEG));
                nnTrainDataset.push_back(trainData);
                
                cerr << "nn example Sn : " << detector->nNClassifier.calcSN(trainData.first) << " Sr : " << detector->nNClassifier.calcSr(trainData.first) << " Sc : " << detector->nNClassifier.calcSc(trainData.first) << endl;
            }
        }
    }
    
    // N-expert - RF
    int nCountRF = 0;
    for(int i = 0; i < detector->scanBBs.size(); i++)
    {
        TYPE_DETECTOR_SCANBB &sbb = detector->scanBBs[i];
        
        //if(sbb.status != DETECTOR_REJECT_VAR)
        //{
            if(sbb.second < LEARNER_TH_OL && detector->rFClassifier.getPosteriors(imgB(sbb.first)) >= 0.1)
            {
                nCountRF++;
                
                TYPE_TRAIN_DATA trainData(make_pair(imgB(sbb.first), CLASS_NEG));
                rfTrainDataset.push_back(trainData);
            }
        //}
    }
    
//    for(int i = 1; i < rfTrainDataset.size(); i++)
//    {
//        int r = (float)theRNG() * i;
//        swap(rfTrainDataset[i], rfTrainDataset[r]);
//    }
    
    cerr << "Generated " << nCountNN << " NN negative samples and " << nCountRF << " RF negative test samples." << endl;
    
    detector->update();
    cerr << "Updated detector" << endl;
    //detector->rFClassifier.train(rfTrainDataset);
    //detector->nNClassifier.train(nnTrainDataset);
    
    detector->nNClassifier.showModel();
}