//
//  NNClassifier.cpp
//  TLD
//
//  Created by 陈裕昕 on 11/4/14.
//  Copyright (c) 2014 Fudan. All rights reserved.
//

#include "NNClassifier.h"

NNClassifier::NNClassifier():
thPos(NN_TH_POS)
{
    
}

NNClassifier::~NNClassifier()
{
    
}

float NNClassifier::calcNCC(const Mat &patch1, const Mat &patch2)
{
    if(NCC_USE_OPENCV)
    {
        Mat nccMat;
        matchTemplate(patch1, patch2, nccMat, CV_TM_CCORR_NORMED);
        
        return nccMat.at<float>(0);
    }
    else
    {
        Mat v0, v1; // convert image to 1 dimension vector
        
        patch1.convertTo(v0, CV_32F);
        patch2.convertTo(v1, CV_32F);
        
        v0 = v0.reshape(0, v0.cols * v0.rows);
        v1 = v1.reshape(0, v1.cols * v1.rows);
        
        Mat v01 = v0.t() * v1;
        
        float norm0, norm1;
        
        norm0 = norm(v0);
        norm1 = norm(v1);
        
        // should not add "abs"
        return v01.at<float>(0) / norm0 / norm1;
    }

}

float NNClassifier::calcSP(const Mat &img)
{
    Mat patch = getPatch(img);
    float maxS = 0;
    for(auto &pPatch : pPatches)
    {
        float S = (calcNCC(pPatch, patch) + 1) * 0.5;
        maxS = max(maxS, S);
    }
    
    return maxS;
}

float NNClassifier::calcSPHalf(const Mat &img)
{
    Mat patch = getPatch(img);
    int count = 0;
    int halfSize = ((int)pPatches.size() + 1) / 2; // complexity of list.size() in C++98 is up to linear.
    float maxS = 0;
    
    for(auto &pPatch : pPatches)
    {
        float S = (calcNCC(pPatch, patch) + 1) * 0.5;
        maxS = max(maxS, S);
        
        if(++count >= halfSize) break;
    }
    
    return maxS;
}

float NNClassifier::calcSN(const Mat &img)
{
    Mat patch = getPatch(img);
    float maxS = 0;
    for(auto &nPatch : nPatches)
    {
        float S = (calcNCC(nPatch, patch) + 1) * 0.5;

        maxS = max(maxS, S);
    }
    
    return maxS;
}

float NNClassifier::calcSr(const Mat &img)
{
    if(pPatches.size() == 0) return 0.;
    if(nPatches.size() == 0) return 1.;
    
    float SP = calcSP(img);
    float SN = calcSN(img);
    
    float dSP = 1 - SP;
    float dSN = 1 - SN;
    
    return dSN / (dSP + dSN);
}

float NNClassifier::calcSc(const Mat &img)
{
    if(pPatches.size() == 0) return 0.;
    if(nPatches.size() == 0) return 1.;
    
    float SPHalf = calcSPHalf(img);
    float SN = calcSN(img);
    
    float dSPHalf = 1 - SPHalf;
    float dSN = 1 - SN;
    
    return dSN / (dSPHalf + dSN);
}

bool NNClassifier::update(const Mat &patch, int c)
{
    float Sr = calcSr(patch);
    float margin = Sr - thPos;
    
    if(c == CLASS_POS && margin < NN_MARGIN)
    {
        if(pPatches.size() >= NN_MODEL_SIZE)
        {
            int idx = (float)theRNG() * pPatches.size();
            pPatches.erase(pPatches.begin() + idx);
        }
        pPatches.push_back(patch);
        
        return true;
    }
    
    if(c == CLASS_NEG && Sr > 0.5) /// think more
    {
        if(nPatches.size() >= NN_MODEL_SIZE)
        {
            int idx = (float)theRNG() * nPatches.size();
            nPatches.erase(nPatches.begin() + idx);
        }
        nPatches.push_back(patch);
        
        return true;
    }
    
    return false;
}

void NNClassifier::addToNewSamples(const Mat &patch, const int c)
{
    const int newSamplesNum = 100;
    
    Mat &samples = (c == CLASS_POS ? newSamplesP : newSamplesN);
    int n = samples.cols / NN_PATCH_SIZE;

    Mat tmp;
    if(n > newSamplesNum)
    {
        tmp = Mat(NN_PATCH_SIZE, samples.cols, CV_8UC3);
        
        samples.colRange(NN_PATCH_SIZE, samples.cols).copyTo(tmp.colRange(0, samples.cols - NN_PATCH_SIZE));
    }
    else
    {
        tmp = Mat(NN_PATCH_SIZE, samples.cols + NN_PATCH_SIZE, CV_8UC3);
        
        if(samples.cols) samples.colRange(0, samples.cols).copyTo(tmp.colRange(0, samples.cols));
    }
    
    samples = tmp;
    
    Mat _patch = patch.clone();
    normalize(_patch, _patch, 0, 255, NORM_MINMAX);
    _patch.convertTo(_patch, CV_8U);
    cvtColor(_patch, samples.colRange(samples.cols - NN_PATCH_SIZE, samples.cols), CV_GRAY2BGR);
}

void NNClassifier::train(const TYPE_TRAIN_DATA_SET &trainDataSet)
{
    for(auto &trainData : trainDataSet)
    {
        Mat patch = getPatch(trainData.first);
        
        if(trainData.second == CLASS_POS || trainData.second == CLASS_NEG)
        {
            if(update(patch, trainData.second))
            {
                addToNewSamples(patch, trainData.second);
            }
        }
    }
    
    for(auto &trainData : trainDataSet)
    {
        if(trainData.second == CLASS_TEST_NEG)
        {
            float Sr = calcSr(trainData.first);
            if(Sr > thPos)
            {
                thPos = Sr;
                cerr << "Increase NN thPos to " << thPos << endl;
            }
        }
    }
}
    
Mat NNClassifier::getPatch(const Mat &img)
{
    Mat patch;
    
    resize(img, patch, Size(NN_PATCH_SIZE, NN_PATCH_SIZE));
    
    Scalar mean, stddev;
    
    meanStdDev(patch, mean, stddev);
    patch.convertTo(patch, CV_32F);
    patch -= mean.val[0];   // think more

    return patch;
}

bool NNClassifier::getClass(const Mat &img)
{
    float Sr = calcSr(img);
    
    return  Sr > thPos ? CLASS_POS : CLASS_NEG;
}

void NNClassifier::showModel()
{
    cerr << "NN Positive samples : " << pPatches.size() << " Negative samples : " << nPatches.size() << endl;
    
    if(newSamplesP.cols) imshow("new positive samples", newSamplesP);
    if(newSamplesN.cols) imshow("new negative samples", newSamplesN);

    waitKey(1);
}