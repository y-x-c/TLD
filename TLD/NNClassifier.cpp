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
        if(NCC_FAST)
        {
            Mat v0, v1;
            
            v0 = patch1.reshape(0, 1);
            v1 = patch2.reshape(0, 1);
            
            double norm0 = 0., norm1 = 0., v0v1 = 0.;
            
            float *data0 = v0.ptr<float>(0);
            float *data1 = v1.ptr<float>(0);
            
            for(int i = 0; i < v0.cols; i++)
            {
                norm0 += data0[i] * data0[i];
                norm1 += data1[i] * data1[i];
                v0v1 += data0[i] * data1[i];
            }
        
            norm0 = sqrt(norm0);
            norm1 = sqrt(norm1);
            
            double ret = v0v1 / norm0 / norm1;
            return ret;
        }
        else
        {
            Mat v0, v1; // convert image to 1 dimension vector
            
            v0 = patch1;
            v1 = patch2;
            
            v0 = v0.reshape(0, v0.cols * v0.rows);
            v1 = v1.reshape(0, v1.cols * v1.rows);
            
            Mat v01 = v0.t() * v1;
            
            float norm0, norm1;
            
            norm0 = norm(v0);
            norm1 = norm(v1);
            
            // should not add "abs"
            float ret = v01.at<float>(0) / norm0 / norm1;
            
            return ret;
        }
    }
}

void NNClassifier::getS(const cv::Mat &img, float &Sp, float &Sn, float &Sr, float &Sc, int &maxSPIdx)
{
    Mat patch = getPatch(img);
    float maxSp = 0, maxSpHalf = 0.;
    
    int halfSize = ((int)pPatches.size() + 1) / 2;
    int count = 0;
    
    maxSPIdx = -1;
    
    for(auto &pPatch : pPatches)
    {
        float S = (calcNCC(pPatch, patch) + 1) * 0.5;

        if(S > maxSp)
        {
            maxSp = S;
            maxSPIdx = count;
        }
        
        if(++count <= halfSize) maxSpHalf = max(maxSpHalf, S);
    }
    
    float maxSn = 0;
    for(auto &nPatch : nPatches)
    {
        float S = (calcNCC(nPatch, patch) + 1) * 0.5;
        maxSn = max(maxSn, S);
    }
    
    float dSpHalf = 1 - maxSpHalf;
    float dSp = 1 - maxSp;
    float dSn = 1 - maxSn;
    
    Sr = dSn / (dSp + dSn);
    Sc = dSn / (dSpHalf + dSn);
    Sn = maxSn;
    Sp = maxSp;
}

float NNClassifier::calcSr(const Mat &img32F, int &maxSPIdx)
{
    float Sr, dummy;
    getS(img32F, dummy, dummy, Sr, dummy, maxSPIdx);
    return Sr;
}

bool NNClassifier::update(const Mat &patch, int c)
{
    int maxSPIdx;
    float Sr = calcSr(patch, maxSPIdx);
    float margin = Sr - thPos;
    
    if(c == CLASS_POS && margin < NN_MARGIN)
    {
        if(pPatches.size() >= NN_MODEL_SIZE)
        {
            int idx = (float)theRNG() * pPatches.size();
            pPatches.erase(pPatches.begin() + idx);
        }
        pPatches.push_back(patch);
        
        // following method is used in OpenTLD implemented by Zdenek Kalal
        //pPatches.insert(pPatches.begin() + maxSPIdx + 1, patch);
        
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
//    const int newSamplesNum = 100;
    
//    Mat &samples = (c == CLASS_POS ? newSamplesP : newSamplesN);
//    int n = samples.cols / NN_PATCH_SIZE;

//    Mat tmp;
//    if(n > newSamplesNum)
//    {
//        tmp = Mat(NN_PATCH_SIZE, samples.cols, CV_8UC3);
//        
//        samples.colRange(NN_PATCH_SIZE, samples.cols).copyTo(tmp.colRange(0, samples.cols - NN_PATCH_SIZE));
//    }
//    else
//    {
//        tmp = Mat(NN_PATCH_SIZE, samples.cols + NN_PATCH_SIZE, CV_8UC3);
//        
//        if(samples.cols) samples.colRange(0, samples.cols).copyTo(tmp.colRange(0, samples.cols));
//    }
//    
//    samples = tmp;
    
//    Mat _patch = patch.clone();
//    normalize(_patch, _patch, 0, 255, NORM_MINMAX);
//    _patch.convertTo(_patch, CV_8U);
//    cvtColor(_patch, samples.colRange(samples.cols - NN_PATCH_SIZE, samples.cols), CV_GRAY2BGR);
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
            int dummy;
            float Sr = calcSr(trainData.first, dummy);
            if(Sr > thPos)
            {
                thPos = Sr;

                stringstream info;
                info << "Increase threshold of positive class to " << thPos;
                outputInfo("NN", info.str());
            }
        }
    }
}
    
Mat NNClassifier::getPatch(const Mat &img32F)
{
    Mat patch;
    
    resize(img32F, patch, Size(NN_PATCH_SIZE, NN_PATCH_SIZE));
    
    Scalar mean, stddev;
    
    meanStdDev(patch, mean, stddev);
    patch -= mean.val[0];   // to reduce brightness influence

    return patch;
}

bool NNClassifier::getClass(const Mat &img32F, TYPE_DETECTOR_SCANBB &sbb)
{
    getS(img32F, sbb.Sp, sbb.Sn, sbb.Sr, sbb.Sc, sbb.maxSPIdx);
    
    return sbb.Sr > thPos ? CLASS_POS : CLASS_NEG;
}

void NNClassifier::showModel()
{
    if(!SHOW_NEW_NN_SAMPLES) return;
    
    stringstream info;
    info << "Positive samples : " << pPatches.size() << " Negative samples : " << nPatches.size();
    
    outputInfo("NN", info.str());
    
//    if(newSamplesP.cols) imshow("new positive samples", newSamplesP);
//    if(newSamplesN.cols) imshow("new negative samples", newSamplesN);

    waitKey(1);
}