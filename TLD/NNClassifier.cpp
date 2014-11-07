//
//  NNClassifier.cpp
//  TLD
//
//  Created by 陈裕昕 on 11/4/14.
//  Copyright (c) 2014 Fudan. All rights reserved.
//

#include "NNClassifier.h"

NNClassifier::NNClassifier()
{
    
}

NNClassifier::~NNClassifier()
{
    
}

float NNClassifier::calcNCC(const Mat &patch1, const Mat &patch2)
{
    /*Mat v1, v2;
    patch1.copyTo(v1);
    patch2.copyTo(v2);
    
    v1 = v1.reshape(0, v1.rows * v1.cols);
    v2 = v2.reshape(0, v2.rows * v2.cols);

    v1.convertTo(v1, CV_32FC1);
    v2.convertTo(v2, CV_32FC1);
     
    float nV1, nV2;
    Mat v1tv2;
    nV1 = norm(v1);
    nV2 = norm(v2);
    v1tv2 = v1.t() * v2;

    float ncc = v1tv2.at<float>(0) / nV1 / nV2;
    */
    
    Mat nccMat;
    matchTemplate(patch1, patch2, nccMat, CV_TM_CCORR_NORMED);

    // debug
    /*imshow("patch1", patch1);
    imshow("patch2", patch2);
    cerr << nccMat << endl;
    waitKey();
     */
    // end debug
    
    return nccMat.at<float>(0);
}

float NNClassifier::calcSP(const Mat &patch)
{
    float maxS = 0;
    for(auto pPatch : pPatches)
    {
        float S = (calcNCC(pPatch, patch) + 1) * 0.5;
        maxS = max(maxS, S);
    }
    
    return maxS;
}

float NNClassifier::calcSPHalf(const Mat &patch)
{
    int count = 0;
    int halfSize = (int)pPatches.size() / 2; // complexity of list.size() in C++98 is up to linear.
    float maxS = 0;
    
    for(auto pPatch : pPatches)
    {
        float S = (calcNCC(pPatch, patch) + 1) * 0.5;
        maxS = max(maxS, S);
        
        if(++count >= halfSize) break;
    }
    
    return maxS;
}

float NNClassifier::calcSN(const Mat &patch)
{
    float maxS = 0;
    for(auto nPatch : nPatches)
    {
        float S = (calcNCC(nPatch, patch) + 1) * 0.5;
        maxS = max(maxS, S);
    }
    
    return maxS;
}

float NNClassifier::calcSr(const Mat &patch)
{
    float SP = calcSP(patch);
    float SN = calcSN(patch);
    
    float dSP = 1 - SP;
    float dSN = 1 - SN;
    
    return dSN / (dSP + dSN);
}

float NNClassifier::calcSc(const Mat &patch)
{
    float SPHalf = calcSPHalf(patch);
    float SN = calcSN(patch);
    
    float dSPHalf = 1 - SPHalf;
    float dSN = 1 - SN;
    
    return dSN / (dSPHalf + dSN);
}

void NNClassifier::update(const Mat &img, int c)
{
    Mat patch = getPatch(img);
    float margin = calcSr(patch) - thNN;
    
    if(margin < thMargin)
    {
        if(c == cPos)
        {
            if(pPatches.size() >= thModelSize)
            {
                int idx = (float)theRNG() * pPatches.size();
                pPatches.erase(pPatches.begin() + idx);
            }
            pPatches.push_back(patch);
        }
        
        if(c == cNeg)
        {
            if(nPatches.size() >= thModelSize)
            {
                int idx = (float)theRNG() * nPatches.size();
                nPatches.erase(nPatches.begin() + idx);
            }
            nPatches.push_back(patch);
        }
    }
}

void NNClassifier::train(const tTrainDataSet &trainDataSet)
{
    for(auto trainData : trainDataSet)
    {
        Mat patch = getPatch(trainData.first);
        bool c = trainData.second;
        
        if(c == cPos && pPatches.size() < thModelSize) pPatches.push_back(patch);
        if(c == cNeg && nPatches.size() < thModelSize) nPatches.push_back(patch);
    }
    
    int nCount = 0;
    for(auto traindata : trainDataSet)
    {
        Mat patch = getPatch(traindata.first);
        if(traindata.second == cNeg)
        {
            float Sr = calcSr(patch);
            if(Sr > thNN)
            {
                thNN = Sr;
                cerr << "Increase thNN to " << thNN << endl;
                // update(patch, cNeg);
            }
            
            if(++nCount >= thModelSize) break;
        }
    }
}

Mat NNClassifier::getPatch(const Mat &img)
{
    Mat patch;
    
    resize(img, patch, Size(patchSize, patchSize));
    
    if(img.channels() == 3)
        cvtColor(patch, patch, CV_BGR2GRAY);
    
    patch.convertTo(patch, CV_8U);
    
    return patch;
}

bool NNClassifier::getClass(const Mat &img)
{
    Mat patch = getPatch(img);
    float Sr = calcSr(patch);
    //cerr << Sr << endl;
    return  Sr > thNN ? cPos : cNeg;
}

void NNClassifier::showModel()
{
    for(auto patch : pPatches)
    {
        Mat img;
        patch.copyTo(img);
        cvtColor(img, img, CV_GRAY2BGR);
        
        imshow("show positive patches", img);
        waitKey();
    }
    destroyWindow("show positive patches");
    
    for(auto patch : nPatches)
    {
        Mat img;
        patch.copyTo(img);
        cvtColor(img, img, CV_GRAY2BGR);
        
        imshow("show negative patches", img);
        waitKey();
    }
    destroyWindow("show negative patches");
}