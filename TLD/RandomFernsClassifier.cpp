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

void RandomFernsClassifier::init(int _nFerns, int _nLeaves, const vector<float> &scales, int initW, int initH)
{
    thPos = FERN_TH_POS;
    
    nFerns = _nFerns;
    nLeaves = _nLeaves;
    
    for(int i = 0; i < nFerns; i++)
    {
        counter.push_back(FernCounter(pow(2, nLeaves)));
    }
    
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
        
        for(int j = 0; j < nLeaves; j++)
        {
            leaves.push_back(tleave[cnt++]);
        }
        
        ferns.push_back(leaves);
    }
    
    for(int s = 0; s < scales.size(); s++)
    {
        int w = round(initW * scales[s]);
        int h = round(initH * scales[s]);
        
        scalesId[make_pair(w, h)] = s;
        
        cmpPts.push_back(vector<vector<CmpPt> >());
        for(int i = 0; i < nFerns; i++)
        {
            cmpPts[s].push_back(vector<CmpPt>());
            for(int j = 0; j < nLeaves; j++)
            {
                int p1x = ferns[i][j].first.x * w;
                int p1y = ferns[i][j].first.y * h;
                int p2x = ferns[i][j].second.x * w;
                int p2y = ferns[i][j].second.y * h;
                
                cmpPts[s][i].push_back(CmpPt(p1x, p1y, p2x, p2y));
            }
        }
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
                counter[iFern][code].voteP();
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
                counter[iFern][code].voteN();
            }
        }
    }
}

int RandomFernsClassifier::getCode(const Mat &img, int idx)
{
    int code = 0;
    int scaleId;
    
    //assert(scalesId.count(make_pair(img.cols, img.rows)) > 0);
    
    scaleId = scalesId[make_pair(img.cols, img.rows)];
    
    for(int i = 0; i < nLeaves; i++)
    {
//        clock_t st = clock();
//        int p1x = ferns[idx][i].first.x * img.cols;
//        int p1y = ferns[idx][i].first.y * img.rows;
//        int p2x = ferns[idx][i].second.x * img.cols;
//        int p2y = ferns[idx][i].second.y * img.rows;
//        clock_t ed = clock();
//        tclock += ed - st;
        
        int p1x, p1y, p2x, p2y;
        cmpPts[scaleId][idx][i].get(p1x, p1y, p2x, p2y);
        
        // use char instead of int
        char v1 = img.at<char>(p1y, p1x);
        char v2 = img.at<char>(p2y, p2x);
        
        code = (code << 1) | (v1 < v2);
    }
    
    return code;
}

float RandomFernsClassifier::getPosteriors(const Mat &img)
{
    float sumP = 0;
    for(int i = 0; i < nFerns; i++)
    {
        int code;
        code = getCode(img, i);
        
        sumP += counter[i][code].getPosteriors();
    }
    
    float averageP = sumP / nFerns;
    
    return averageP;
}

float RandomFernsClassifier::getSumPosteriors(const Mat &img)
{
    float sumP = 0;
    for(int i = 0; i < nFerns; i++)
    {
        int code;
        code = getCode(img, i);
        
        sumP += counter[i][code].getPosteriors();
    }
    
    return sumP;
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