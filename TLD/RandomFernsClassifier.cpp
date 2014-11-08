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
            leaf.first = Point2f(float(theRNG()), float(theRNG()));
            leaf.second = Point2f(float(theRNG()), float(theRNG()));
            
            leaves.push_back(leaf);
        }
        
        ferns.push_back(leaves);
    }
}

RandomFernsClassifier::~RandomFernsClassifier()
{
    
}

void RandomFernsClassifier::update(const Mat &img, bool c, float p)
{
    // Do gaussian blur in whole image to have a high preformance
    //Mat img;
    //GaussianBlur(_img, img, Size(3, 3), 0);
    
    if(p == -1.) p = getPosteriors(img);
    
    if(c == CLASS_POS)
    {
        if(p <= FERN_TH_POS)
        {
            //static int count = 0;
            //cerr << ++count << endl;
            for(int iFern = 0; iFern < nFerns; iFern++)
            {
                int code = getCode(img, iFern);
                counter[iFern][code].first++;
                //cerr << iFern << " " << code << endl;
            }
        }
    }
    else
    {
        if(p >= FERN_TH_NEG)
        {
            for(int iFern = 0; iFern < nFerns; iFern++)
            {
                int code = getCode(img, iFern);
                counter[iFern][code].second++;
                //cerr << iFern << " " << code << endl;
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
        
        //cerr << (v1 < v2);
    }
    
    //cerr << endl;
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

bool RandomFernsClassifier::getClass(const Mat &img)
{
    // assert : _img.type() == CV_8U
    
    // Do gaussian blur in whole image to have a high preformance
    //Mat img;
    //GaussianBlur(_img, img, Size(3, 3), 0);
    
    if(getPosteriors(img) >= FERN_TH_POS)
        return CLASS_POS;
    else
        return CLASS_NEG;
}

void RandomFernsClassifier::train(const TYPE_TRAIN_DATA_SET &trainDataSet)
{
    for(auto trainData : trainDataSet)
    {
        // assert : trainData.first.type() == CV_8U
        update(trainData.first, trainData.second);
    }
}