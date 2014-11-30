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

void RandomFernsClassifier::gen4Pts(float ox, float oy, vector<TYPE_FERN_LEAF> &tleave)
{
    Point2f origin(ox, oy);
    Point2f t, b, l, r;
    float tmpx, tmpy;
    
    tmpy = oy - (getRNG() + RF_FEA_OFF);
    tmpy = max(0.f, tmpy);
    t = Point2f(ox, tmpy);
    
    tmpy = oy + (getRNG() + RF_FEA_OFF);
    tmpy = min(1.f, tmpy);
    b = Point2f(ox, tmpy);
    
    tmpx = ox - (getRNG() + RF_FEA_OFF);
    tmpx = max(0.f, tmpx);
    l = Point2f(tmpx, oy);
    
    tmpx = ox + (getRNG() + RF_FEA_OFF);
    tmpx = min(1.f, tmpx);
    r = Point2f(tmpx, oy);
    
    tleave.push_back(make_pair(origin, t));
    tleave.push_back(make_pair(origin, b));
    tleave.push_back(make_pair(origin, l));
    tleave.push_back(make_pair(origin, r));

}

void RandomFernsClassifier::init(int _nFerns, int _nLeaves, const vector<float> &scales, const Mat &imgB, int initW, int initH)
{
    thPos = RF_TH_POS;
    
    nFerns = _nFerns;
    nLeaves = _nLeaves;
    
    for(int i = 0; i < nFerns; i++)
    {
        counter.push_back(FernCounter(pow(2, nLeaves)));
    }
    
    vector<TYPE_FERN_LEAF> tleave;
    
    for(float sx = 0; sx < 1; sx += RF_FEA_SHIFT)
    {
        if(!(sx > 0 && sx < 1)) continue;
        
        for(float sy = 0; sy < 1; sy += RF_FEA_SHIFT)
        {
            if(!(sy > 0 && sy < 1)) continue;
            
            gen4Pts(sx, sy, tleave);
        }
    }
    
    for(float sx = RF_FEA_SHIFT / 2; sx < 1; sx += RF_FEA_SHIFT)
    {
        if(!(sx > 0 && sx < 1)) continue;
        
        for(float sy = RF_FEA_SHIFT / 2; sy < 1; sy += RF_FEA_SHIFT)
        {
            if(!(sy > 0 && sy < 1)) continue;
            
            gen4Pts(sx, sy, tleave);
        }
    }
    
    if(RND_SHUFFLE_STD)
    {
        random_shuffle(tleave.begin(), tleave.end());
    }
    else
    {
        for(int i = 0; i < tleave.size(); i++)
        {
            swap(tleave[i], tleave[(float)theRNG() * tleave.size()]);
        }
    }
    
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

    genOffsets(scales, imgB);
}

RandomFernsClassifier::~RandomFernsClassifier()
{
    
}

void RandomFernsClassifier::genOffsets(const vector<float> &scales, const Mat &img)
{
    for(int s = 0; s < scales.size(); s++)
    {
        offsets.push_back(vector<vector<pair<int, int> > >());
        for(int i = 0; i < nFerns; i++)
        {
            offsets[s].push_back(vector<pair<int, int> >());
            for(int j = 0; j < nLeaves; j++)
            {
                int p1x, p1y, p2x, p2y;
                cmpPts[s][i][j].get(p1x, p1y, p2x, p2y);

                int offset1 = p1y * img.step + p1x;
                int offset2 = p2y * img.step + p2x;

                offsets[s][i].push_back(make_pair(offset1, offset2));
            }
        }
    }
}

float RandomFernsClassifier::getRNG()
{
    return (float)theRNG();
}

void RandomFernsClassifier::update(const Mat &img, bool c)
{
    int scaleId = scalesId[make_pair(img.cols, img.rows)];
    int p = getPosteriors(img, scaleId);
    
    if(c == CLASS_POS)
    {
        if(p < thPos)
        {
            for(int iFern = 0; iFern < nFerns; iFern++)
            {
                int code = getCode(img, iFern, scaleId);
                counter[iFern][code].voteP();
            }
        }
    }
    
    if(c == CLASS_NEG)
    {
        if(p >= RF_TH_NEG)
        {
            for(int iFern = 0; iFern < nFerns; iFern++)
            {
                int code = getCode(img, iFern, scaleId);
                counter[iFern][code].voteN();
            }
        }
    }
}

int RandomFernsClassifier::getCode(const Mat &img, int idx, int scaleId)
{
    int code = 0;

    //assert(scalesId.count(make_pair(img.cols, img.rows)) > 0);
    
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
        //int v1 = img.at<uchar>(p1y, p1x);
        //int v2 = img.at<uchar>(p2y, p2x);

        int v1 = *(uchar*)(img.data + offsets[scaleId][idx][i].first);
        int v2 = *(uchar*)(img.data + offsets[scaleId][idx][i].second);
        
        code = (code << 1) | (v1 < v2);
    }
    
    return code;
}

float RandomFernsClassifier::getPosteriors(const Mat &img, int scaleId)
{
    float sumP = 0;
    for(int i = 0; i < nFerns; i++)
    {
        int code;
        code = getCode(img, i, scaleId);
        
        sumP += counter[i][code].getPosteriors();
    }
    
    float averageP = sumP / nFerns;
    
    return averageP;
}

float RandomFernsClassifier::getSumPosteriors(const Mat &img, int scaleId)
{
    float sumP = 0;
    for(int i = 0; i < nFerns; i++)
    {
        int code;
        code = getCode(img, i, scaleId);
        
        sumP += counter[i][code].getPosteriors();
    }
    
    return sumP;
}

bool RandomFernsClassifier::getClass(const Mat &img, TYPE_DETECTOR_SCANBB &sbb)
{
    // assert : _img.type() == CV_8U
    int scaleId = scalesId[make_pair(img.cols, img.rows)];
    
    if((sbb.posterior = getPosteriors(img, scaleId)) >= thPos)
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
            int scaleId = scalesId[make_pair(trainData.first.cols, trainData.first.rows)];
            float p = getPosteriors(trainData.first, scaleId);
            if(thPos < p)
            {
                thPos = p;
                
                stringstream info;
                info << "Increase threshold of positive class to " << thPos;
                outputInfo("RF", info.str());
            }
        }
    }
}
