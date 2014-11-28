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
    
    /*for(int i = 0; i < nFerns; i++)
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
    }*/
    
    vector<TYPE_FERN_LEAF> tleave;
    float SHI = 1. / 5;
    float OFF = SHI;
    
    for(float sx = 0; sx < 1; sx += SHI)
    {
        if(!(sx > 0 && sx < 1)) continue;
        
        for(float sy = 0; sy < 1; sy += SHI)
        {
            if(!(sy > 0 && sy < 1)) continue;
            
            Point2f origin(sx, sy);
            Point2f t, b, l, r;
            float tmpx, tmpy;
            
            tmpy = sy - (getRNG() + OFF);
            tmpy = max(0.f, tmpy);
            t = Point2f(sx, tmpy);
            
            tmpy = sy + (getRNG() + OFF);
            tmpy = min(1.f, tmpy);
            b = Point2f(sx, tmpy);
            
            tmpx = sx - (getRNG() + OFF);
            tmpx = max(0.f, tmpx);
            l = Point2f(tmpx, sy);
            
            tmpx = sx + (getRNG() + OFF);
            tmpx = min(1.f, tmpx);
            r = Point2f(tmpx, sy);
            
            tleave.push_back(make_pair(origin, t));
            tleave.push_back(make_pair(origin, b));
            tleave.push_back(make_pair(origin, l));
            tleave.push_back(make_pair(origin, r));
        }
    }
    
    for(float sx = SHI / 2; sx < 1; sx += SHI)
    {
        if(!(sx > 0 && sx < 1)) continue;
        
        for(float sy = SHI / 2; sy < 1; sy += SHI)
        {
            if(!(sy > 0 && sy < 1)) continue;
            
            Point2f origin(sx, sy);
            Point2f t, b, l, r;
            float tmpx, tmpy;
            
            tmpy = sy - (getRNG() + OFF);
            tmpy = max(0.f, tmpy);
            t = Point2f(sx, tmpy);
            
            tmpy = sy + (getRNG() + OFF);
            tmpy = min(1.f, tmpy);
            b = Point2f(sx, tmpy);
            
            tmpx = sx - (getRNG() + OFF);
            tmpx = max(0.f, tmpx);
            l = Point2f(tmpx, sy);
            
            tmpx = sx + (getRNG() + OFF);
            tmpx = min(1.f, tmpx);
            r = Point2f(tmpx, sy);
            
            tleave.push_back(make_pair(origin, t));
            tleave.push_back(make_pair(origin, b));
            tleave.push_back(make_pair(origin, l));
            tleave.push_back(make_pair(origin, r));
        }
    }
    
    random_shuffle(tleave.begin(), tleave.end());
    
    int cnt = 0;
    for(int i = 0; i < nFerns; i++)
    {
        vector<TYPE_FERN_LEAF> leaves;
        
        //Mat ou(50, 50, CV_8UC3, Scalar::all(0));
        
        for(int j = 0; j < nLeaves; j++)
        {
            //line(ou, 50 * tleave[cnt].first, 50 * tleave[cnt].second, Scalar::all(255));
            leaves.push_back(tleave[cnt++]);
        }
        
        //imshow("features", ou);
        //waitKey();
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
    for(int b = 0; b < 2; b++) // bootstrap
    {
        for(auto &trainData : trainDataSet)
        {
            // assert : trainData.first.type() == CV_8U
            if(trainData.second == CLASS_POS || trainData.second == CLASS_NEG)
                update(trainData.first, trainData.second);
        }
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