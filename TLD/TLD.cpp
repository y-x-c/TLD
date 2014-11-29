//
//  TLD.cpp
//  TLD
//
//  Created by 陈裕昕 on 14/11/7.
//  Copyright (c) 2014年 Fudan. All rights reserved.
//

#include "TLD.h"

TLD::TLD(const Mat &img, const TYPE_BBOX &_bb)
{
    bbox = _bb;
    
    setNextFrame(img);
    
    detector.init(nextImg, nextImgB, nextImg32F, _bb);
    learner.init(&detector);
    
    trainValid = true;
}

TLD::~TLD()
{
    
}

void TLD::setNextFrame(const cv::Mat &frame)
{
    cv::swap(prevImg, nextImg);
    
    cvtColor(frame, nextImg, CV_BGR2GRAY);
    
    GaussianBlur(nextImg, nextImgB, Size(3, 3), 1.5);
    
    nextImg.convertTo(nextImg32F, CV_32F);
}

TYPE_BBOX TLD::getInside(const TYPE_BBOX &bb)
{
    int tlx = max(0, bb.tl().x), tly = max(0, bb.tl().y);
    int brx = min(nextImg.cols, bb.br().x), bry = min(nextImg.rows, bb.br().y);
    Rect retBB(tlx, tly, brx - tlx, bry - tly);
    
    if(retBB.area() <= 0) retBB = BB_ERROR;
    return retBB;
}

int TLD::cluster()
{
    clusterBB.clear();
    
    vector<vector<int> > edges(detectorRet.size());
    
    for(int i = 0; i < detectorRet.size(); i++)
    {
        for(int j = i + 1; j < detectorRet.size(); j++)
        {
            if(overlap(detectorRet[i], detectorRet[j]) > 0.5)
            {
                edges[i].push_back(j);
                edges[j].push_back(i);
            }
        }
    }
    
    vector<int> belong(detectorRet.size(), -1);
    queue<int> Q;
    
    int cntBelong = 0;
    for(int i = 0; i < detectorRet.size(); i++)
    {
        if(belong[i] != -1) continue;
        
        belong[i] = cntBelong;
        Q.push(i);
        
        while(!Q.empty())
        {
            int x = Q.front();
            Q.pop();
            for(auto &y : edges[x])
            {
                if(belong[y] == -1)
                {
                    belong[y] = cntBelong;
                    Q.push(y);
                }
            }
        }
        
        cntBelong++;
    }
    
    for(int i = 0; i < cntBelong; i++)
    {
        float x = 0., y = 0., height = 0., width = 0.;
        float Sc = 0.;
        int count = 0;
        
        for(int j = 0; j < detectorRet.size(); j++)
        {
            if(belong[j] != i) continue;
            
            x += detectorRet[j].x;
            y += detectorRet[j].y;
            height += detectorRet[j].height;
            width += detectorRet[j].width;
            
            Sc += detectorRet[j].Sc;
            
            count++;
        }
        
        x /= count;
        y /= count;
        height /= count;
        width /= count;
        Sc /= count;
        
        TYPE_DETECTOR_SCANBB _rect(Rect(round(x), round(y), round(width), round(height)));
        _rect.Sc = Sc;
        clusterBB.push_back(_rect);
    }
    
    stringstream info;
    info << "Found " << cntBelong << " clusters.";
    outputInfo("TLD", info.str());
    
    return cntBelong;
}

int TLD::track()
{
    //track
    tracker = new MedianFlow(prevImg, nextImg);
    
    int trackerStatus;
    TYPE_MF_BB _trackerRet = tracker->trackBox(bbox, trackerStatus);
    TYPE_DETECTOR_SCANBB trackerRet(Rect(round(_trackerRet.x), round(_trackerRet.y), round(_trackerRet.width), round(_trackerRet.height)));
    TYPE_DETECTOR_SCANBB trackerRetInside = getInside(trackerRet);
    
    if(trackerStatus == MF_TRACK_SUCCESS)
    {
        detector.updataNNPara(nextImg32F, trackerRetInside);
    }
    
    //detect
    detector.dectect(nextImg, nextImgB, nextImg32F, detectorRet);
    
    //integrate
    float trackSc = -1;
    TYPE_DETECTOR_SCANBB finalBB, finalBBInside;
    
    if(trackerStatus != MF_TRACK_SUCCESS && detectorRet.size() == 0)
    {
        outputInfo("TLD", "Not visible.");
        bbox = BB_ERROR;
        
        delete tracker;
        return TLD_TRACK_FAILED;
    }
    
    if(trackerStatus != MF_TRACK_SUCCESS)
    {
        trainValid = false;
    }
    else
    {
        int value = trackerRetInside.Sr > max(0.7f, detector.getNNThPos());
        
        if(!trainValid && value) outputInfo("TLD", "Trust current trajectory.");
        
        trainValid |= value;
    }
    
    if(trackerStatus == MF_TRACK_SUCCESS)
    {
        trackSc = trackerRetInside.Sc;
        finalBB = trackerRet;
        finalBBInside = trackerRetInside;
    }
    
    if(detectorRet.size())
    {
        //cluster
        int cntBelong = cluster();
        
        if(trackerStatus == MF_TRACK_SUCCESS)
        {
            int confidentDetections = 0;
            int lastId = 0;
            
            for(int i = 0; i < cntBelong; i++)
            {
                if(overlap(clusterBB[i], trackerRet) < 0.5 && clusterBB[i].Sc > trackSc)
                {
                    confidentDetections++;
                    lastId = i;
                }
            }
            
            if(confidentDetections == 1)
            {
                outputInfo("TLD", "Found a better match.. Reinitialize bounding box.");
                
                finalBB = finalBBInside = clusterBB[lastId];
                
                trainValid = false;
            }
            else
            {
                stringstream info;
                info << "Found " << confidentDetections << " confidence clusters. Adjust bouding box by detection result.";
                outputInfo("TLD", info.str());

                int closeDetections = 0;
                float cx = 0., cy = 0., cw = 0., ch = 0.;

                for(int i = 0; i < detectorRet.size(); i++)
                {
                    if(overlap(detectorRet[i], trackerRet) > 0.7)
                    {
                        closeDetections++;
                        
                        cx += detectorRet[i].x;
                        cy += detectorRet[i].y;
                        cw += detectorRet[i].width;
                        ch += detectorRet[i].height;
                    }
                }
                
                int tWeight = 10;
                cx = (tWeight * trackerRet.x + cx) / (tWeight + closeDetections);
                cy = (tWeight * trackerRet.y + cy) / (tWeight + closeDetections);
                cw = (tWeight * trackerRet.width + cw) / (tWeight + closeDetections);
                ch = (tWeight * trackerRet.height + ch) / (tWeight + closeDetections);
                
                finalBB = Rect(round(cx), round(cy), round(cw), round(ch));
                finalBBInside = getInside(finalBB);
                
                if(finalBBInside.area() <= 0)
                {
                    outputInfo("TLD", "Not visable.");
                    
                    trainValid = false;
                    return TLD_TRACK_FAILED;
                }
            }
        }
        else
        {
            if(cntBelong == 1)
            {
                outputInfo("TLD", "Tracking failed but found only one cluster. Reinitialize.");
                finalBB = finalBBInside = clusterBB[0];
                
                trainValid = false;
            }
            else
            {
                outputInfo("TLD", "Tracking failed and found too many clusters, discard detection results.");
                outputInfo("TLD", "Not visable.");
                bbox = BB_ERROR;
                
                trainValid = false;
                
                delete tracker;
                return TLD_TRACK_FAILED;
            }
        }
    }
    
    detector.updataNNPara(nextImg32F, finalBBInside);
    
    if(trainValid)
    {
        if(finalBB == finalBBInside)
        {
            if(finalBBInside.Sr > 0.5)
            {
                if(finalBBInside.Sn < 0.95)
                {
                    learner.learn(nextImg, nextImgB, nextImg32F, finalBB);
                }
                else
                {
                    outputInfo("TLD", "Probably already in negative example set, not learning");
                }
            }
            else
            {
                //trainValid = 0;
                outputInfo("TLD", "Changing too fast, not learning");
            }
        }
        else
        {
            outputInfo("TLD", "Final result is out of image, not learning");
        }
    }
    else
    {
        outputInfo("TLD", "Final result is not in a convinced trajectory, not training");
    }
    
    bbox = finalBB;

    delete tracker;
    return TLD_TRACK_SUCCESS;
}

TYPE_BBOX TLD::getBB()
{
    return bbox;
}

float TLD::overlap(const TYPE_BBOX &bb1, const TYPE_BBOX &bb2)
{
    int tlx, tly, brx, bry;
    
    tlx = max(bb1.tl().x, bb2.tl().x);
    tly = max(bb1.tl().y, bb2.tl().y);
    brx = min(bb1.br().x, bb2.br().x);
    bry = min(bb1.br().y, bb2.br().y);
    
    if(tlx > brx) return 0.;
    if(tly > bry) return 0.;
    
    float area_n = (brx - tlx) * (bry - tly);
    float area_u = bb1.area() + bb2.area() - area_n;
    
    return area_n / area_u;
}