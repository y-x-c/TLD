//
//  Learner.cpp
//  TLD
//
//  Created by 陈裕昕 on 14/11/7.
//  Copyright (c) 2014年 Fudan. All rights reserved.
//

#include "Learner.h"

void Learner::init(Detector *_detector)
{
    detector = _detector;
    
    patchGenerator = PatchGenerator(0, 0, DETECTOR_UPDATE_WARP_NOISE, DETECTOR_UPDATE_WARP_BLUR, 1. - DETECTOR_UPDATE_WARP_SCALE, 1. + DETECTOR_UPDATE_WARP_SCALE, -DETECTOR_UPDATE_WARP_ANGLE, DETECTOR_UPDATE_WARP_ANGLE, -DETECTOR_UPDATE_WARP_ANGLE, DETECTOR_UPDATE_WARP_ANGLE);
}

Learner::~Learner()
{
    
}

void Learner::learn(const Mat &img, const Mat &imgB, const Mat &img32F, const TYPE_BBOX &ret)
{
    TYPE_TRAIN_DATA_SET &nnTrainDataset = detector->trainDataSetNN;
    nnTrainDataset.clear();
    TYPE_TRAIN_DATA_SET &rfTrainDataset = detector->trainDataSetRF;
    rfTrainDataset.clear();
    auto &scanBBs = detector->scanBBs;
    
    detector->sortByOverlap(ret, true);
    
    int count = 0;
    
    // P-expert - NN
    nnTrainDataset.push_back(make_pair(img32F(scanBBs[0]), CLASS_POS));
    
    // P-expert - RF
    int tlx = img.cols, tly = img.rows, brx = 0, bry = 0;
    
    for(int i = 0; i < LEARNER_N_GOOD_BB && scanBBs[i].overlap >= GOODBB_OL; i++)
    {
        tlx = min(tlx, scanBBs[i].tl().x);
        tly = min(tly, scanBBs[i].tl().y);
        brx = max(brx, scanBBs[i].br().x);
        bry = max(bry, scanBBs[i].br().y);
    }
    
    Point tl(tlx, tly), br(brx, bry);
    Rect bbHull(tl, br);
    
    int cx, cy;
    cx = round((double)(tlx + brx) / 2);
    cy = round((double)(tly + bry) / 2);
    
    for(int j = 0; j < LEARNER_N_WARPED; j++)
    {
        Mat warped;
        
        if(j != 0)
        {
            patchGenerator(imgB, Point(cx, cy), warped, bbHull.size(), theRNG());
            // for optimum in RF::getcode()
            Mat tmp(imgB.size() ,CV_8U, Scalar::all(0));
            warped.copyTo(tmp(Rect(0, 0, bbHull.size().width, bbHull.size().height)));
            warped = tmp;
        }
        else
        {
            warped = imgB(bbHull);
        }
        
        for(int i = 0; i < LEARNER_N_GOOD_BB && scanBBs[i].overlap >= GOODBB_OL; i++)
        {
            Rect rect(scanBBs[i].tl() - tl, scanBBs[i].br() - tl);
            
            TYPE_TRAIN_DATA trainData(make_pair(warped(rect), CLASS_POS));
            rfTrainDataset.push_back(trainData);
            
            count++;
        }
    }

    // N-expert - NN
    int nCountNN = 0;
    for(int i = 0; i < detector->scanBBs.size(); i++)
    {
        TYPE_DETECTOR_SCANBB &sbb = detector->scanBBs[i];
        
        if(sbb.status != DETECTOR_REJECT_VAR)
        {
            if(sbb.overlap < BADBB_OL)
            {
                nCountNN++;
                
                TYPE_TRAIN_DATA trainData(make_pair(img32F(sbb), CLASS_NEG));
                nnTrainDataset.push_back(trainData);
            }
        }
        
        if(nCountNN == LEARNER_N_NN_NEG) break;
    }
    
    // N-expert - RF
    int nCountRF = 0;
    for(int i = 0; i < detector->scanBBs.size(); i++)
    {
        TYPE_DETECTOR_SCANBB &sbb = detector->scanBBs[i];
        
        if(sbb.status != DETECTOR_REJECT_VAR)
        {
            if(sbb.overlap < BADBB_OL && sbb.posterior >= 0.1)
            {
                nCountRF++;
                
                TYPE_TRAIN_DATA trainData(make_pair(imgB(sbb), CLASS_NEG));
                rfTrainDataset.push_back(trainData);
            }
        }
    }
    
    stringstream info;
    info << "Generated 1 positive example for NN, and " << count << " positive example for RF.";
    outputInfo("Learner", info.str());
    
    stringstream info2;
    info2 << "Generated " << nCountNN << " NN negative sample(s), and " << nCountRF << " RF negative example(s).";
    outputInfo("Learner", info2.str());
    
    detector->update();
    stringstream info3;
    info3 << "Updated detector.";
    outputInfo("Learner", info3.str());
    
    detector->nNClassifier.showModel();
}
