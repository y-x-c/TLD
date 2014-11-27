//
//  RandomFernsClassifier.cpp
//  TLD
//
//  Created by 陈裕昕 on 11/4/14.
//  Copyright (c) 2014 Fudan. All rights reserved.
//

#include "RandomFernsClassifier.h"

RandomFernsClassifier::RandomFernsClassifier()
{
    
}

RandomFernsClassifier::RandomFernsClassifier(int _nFerns, int _nLeaves)
{
    thPos = FERN_TH_POS;
    
    nFerns = _nFerns;
    nLeaves = _nLeaves;
    
    for(int i = 0; i < nFerns; i++)
    {
        counter.push_back(TYPE_FERN_PNCOUNTER(pow(2, nLeaves), make_pair(0, 0)));
    }
    
    for(int i = 0; i < nFerns; i++)
    {
        vector<TYPE_FERN_LEAF> leaves;
        for(int j = 0; j < nLeaves; j++)
        {
            TYPE_FERN_LEAF leaf;
            leaf.first = Point2f(getRNG(), getRNG());
            leaf.second = Point2f(getRNG(), getRNG());
            
            leaves.push_back(leaf);
        }
        
        ferns.push_back(leaves);
    }
}

RandomFernsClassifier::~RandomFernsClassifier()
{
    
}

float RandomFernsClassifier::getRNG()
{
    return (float)theRNG();
}

void RandomFernsClassifier::update(const Mat &img, bool c, float p)
{
    if(p == -1.) p = getPosteriors(img);
    
    if(c == CLASS_POS)
    {
        if(p < thPos)
        {
            for(int iFern = 0; iFern < nFerns; iFern++)
            {
                int code = getCode(img, iFern);
                counter[iFern][code].first++;
            }
        }
    }
    
    if(c == CLASS_NEG)
    {
        //if(p >= FERN_TH_POS)
        if(p >= 0.5)
        {
            for(int iFern = 0; iFern < nFerns; iFern++)
            {
                int code = getCode(img, iFern);
                counter[iFern][code].second++;
            }
        }
    }
}

int RandomFernsClassifier::getCode(const Mat &img, int idx)
{
    int code = 0;
    for(int i = 0; i < nLeaves; i++)
    {
        int p1x = ferns[idx][i].first.x * img.cols;
        int p1y = ferns[idx][i].first.y * img.rows;
        int p2x = ferns[idx][i].second.x * img.cols;
        int p2y = ferns[idx][i].second.y * img.rows;
        
        // use char instead of int
        int v1 = img.at<char>(p1y, p1x);
        int v2 = img.at<char>(p2y, p2x);
        
        code = (code << 1) | (v1 < v2);
    }
    
    return code;
}

float RandomFernsClassifier::getPosteriors(const Mat &img)
{
    float sumP = 0;
    for(int i = 0; i < nFerns; i++)
    {
        int code, p, n;
        code = getCode(img, i);
        p = counter[i][code].first;
        n = counter[i][code].second;
        
        if(p == 0)
            ;
            //sumP += 0.0;
        else
            sumP += (float)p / (p + n);
    }
    
    float averageP = sumP / nFerns;
    
    return averageP;
}

bool RandomFernsClassifier::getClass(const Mat &img, TYPE_DETECTOR_SCANBB &sbb)
{
    // assert : _img.type() == CV_8U
    
    if((sbb.posterior = getPosteriors(img)) >= thPos)
        return CLASS_POS;
    else
        return CLASS_NEG;
}

void RandomFernsClassifier::train(const TYPE_TRAIN_DATA_SET &trainDataSet)
{
    for(auto &trainData : trainDataSet)
    {
        // assert : trainData.first.type() == CV_8U
        if(trainData.second == CLASS_POS || trainData.second == CLASS_NEG)
            update(trainData.first, trainData.second);
    }
    
    for(auto &trainData : trainDataSet)
    {
        if(trainData.second == CLASS_TEST_NEG)
        {
            float p = getPosteriors(trainData.first);
            if(thPos < p)
            {
                thPos = p;
                cerr << "Increase RF thPos to " << thPos << endl;
            }
        }
    }
}