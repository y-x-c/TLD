//
//  TLD.cpp
//  TLD
//
//  Created by 陈裕昕 on 14/11/7.
//  Copyright (c) 2014年 Fudan. All rights reserved.
//

#include "TLD.h"

TLD::TLD(const Mat &img, const Rect &_bb):
bb(_bb), valid(false)
{
    setNextFrame(img);
    
    detector = Detector(nextImg, nextImgB, _bb);
    learner = Learner(&detector);
}

TLD::~TLD()
{
    
}

void TLD::setNextFrame(const cv::Mat &frame)
{
    cv::swap(prevImg, nextImg);
    
    cvtColor(frame, nextImg, CV_BGR2GRAY);
    
    GaussianBlur(nextImg, nextImgB, Size(3, 3), 1.5);
}

Rect TLD::getInside(const Rect &bb)
{
    int tlx = max(0, bb.tl().x), tly = max(0, bb.tl().y);
    int brx = min(nextImg.cols, bb.br().x), bry = min(nextImg.rows, bb.br().y);
    Rect retBB(tlx, tly, brx - tlx, bry - tly);
    
    return retBB;
}

void TLD::track(Rect &bbTrack, TYPE_DETECTOR_RET &bbDetect)
{
    ///// debug
    bbTrack = BB_ERROR;
    bbDetect.clear();
    /////
    
    tracker = new MedianFlow(prevImg, nextImg);
    
    //track
    int trackerStatus;
    TYPE_MF_BB _trackerRet = tracker->trackBox(bb, trackerStatus);
    TYPE_DETECTOR_SCANBB trackerRet(Rect(cvRound(_trackerRet.x), cvRound(_trackerRet.y), cvRound(_trackerRet.width), cvRound(_trackerRet.height)));
    TYPE_DETECTOR_SCANBB trackerRetInside = getInside(trackerRet);

    if(trackerRetInside.area() <= 0) trackerStatus = MF_TRACK_F_BOX;
    
    if(trackerStatus == MF_TRACK_SUCCESS)
    {
        detector.updataNNPara(nextImg, trackerRetInside);
    }
    
    //detect
    TYPE_DETECTOR_RET detectorRet;
    detector.dectect(nextImg, nextImgB, detectorRet);
    
    //integrate
    float trackSc = -1;
    TYPE_DETECTOR_SCANBB finalBB, finalBBInside;
    
    if(trackerStatus != MF_TRACK_SUCCESS && detectorRet.size() == 0)
    {
        cerr << "Not visible." << endl;
        bb = BB_ERROR;
        
        delete tracker;
        return;
    }
    
    if(trackerStatus != MF_TRACK_SUCCESS)
    {
        valid = false;
    }
    else
    {
        valid |= trackerRetInside.Sc > max(0.7f, detector.getNNThPos());
    }
    
    if(trackerStatus == MF_TRACK_SUCCESS)
    {
        cerr << "Track bb : " << trackerRet << " Sc : " << trackerRetInside.Sc << " Sr : " << trackerRetInside.Sr << "  Sn : " << trackerRetInside.Sn << " Sp: " << trackerRetInside.Sp << endl;

        trackSc = trackerRetInside.Sc;
        finalBB = trackerRet;
        finalBBInside = trackerRetInside;
        
        // debug
        bbTrack = trackerRet;
        // end debug
    }
    
    if(detectorRet.size())
    {
        //debug
        bbDetect = detectorRet;
        //end debug
        
        //cluster
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
        
        TYPE_DETECTOR_RET clusterBB;
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
            
            TYPE_DETECTOR_SCANBB _rect(Rect(cvRound(x), cvRound(y), cvRound(width), cvRound(height)));
            _rect.Sc = Sc;
            clusterBB.push_back(_rect);
            
            cerr << "Cluster " << i << " : " << _rect  << " Sc : " << Sc << endl;
        }
        
        cout << "Found " << cntBelong << " clusters." << endl;
        
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
                cerr << "Found a better match.. Regard it as the final result.." << endl;
                finalBB = finalBBInside = clusterBB[lastId];
                
                valid = false;
            }
            else
            {
                cerr << "Found " << confidentDetections << " confidence clusters. Mix track bb and detection bbs which are close to track bb..." << endl;

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
                
                finalBB = Rect(cvRound(cx), cvRound(cy), cvRound(cw), cvRound(ch));
                finalBBInside = getInside(finalBB);
            }
        }
        else
        {
            if(cntBelong == 1)
            {
                cerr << "Track failed but found only one cluster, regard it as the final result." << endl;
                finalBB = finalBBInside = clusterBB[0];
                
                valid = false;
            }
            else
            {
                cerr << "Track failed and found too many clusters, discard detection result." << endl;
                cerr << "Not visible." << endl;
                bb = BB_ERROR;
                
                valid = false;
                
                delete tracker;
                return;
            }
        }
    }
    
    detector.updataNNPara(nextImg, finalBBInside);
    
    cerr << "Final bb : " << finalBB << " Sc : " << finalBBInside.Sc << " Sr : " << finalBBInside.Sr << "  Sn : " << finalBBInside.Sn << " Sp: " << finalBBInside.Sp << endl;
    
    if(valid)
    {
        if(finalBB == finalBBInside)
        {
            if(finalBBInside.Sn < 0.95 && finalBBInside.Sr > 0.5)
            {
                learner.learn(nextImg, nextImgB, finalBB);
            }
            else
            {
                //valid = 0;
                cerr << "changing too fast, not learning" << endl;
            }
        }
        else
        {
            cerr << "finalbb is out of image, not learning" << endl;
        }
    }
    else
    {
        cerr << "Final result is not in a convinced trajectory, not training" << endl;
    }
    
    bb = finalBB;

    delete tracker;
}

Rect TLD::getBB()
{
    return bb;
}

float TLD::overlap(const TYPE_DETECTOR_BB &bb1, const TYPE_DETECTOR_BB &bb2)
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