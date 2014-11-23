//
//  TLD.cpp
//  TLD
//
//  Created by 陈裕昕 on 14/11/7.
//  Copyright (c) 2014年 Fudan. All rights reserved.
//

#include "TLD.h"

TLD::TLD(const Mat &img, const Rect &_bb):
bb(_bb)
{
    setNextFrame(img);
    
    detector = Detector(nextImg, _bb);
    learner = Learner(&detector);
}

TLD::~TLD()
{
    
}

void TLD::setNextFrame(const cv::Mat &frame)
{
    cv::swap(prevImg, nextImg);
    
    cvtColor(frame, nextImg, CV_BGR2GRAY);
    GaussianBlur(nextImg, nextImg, Size(9, 9), 1.5);
}

Rect TLD::getInside(const Rect &bb)
{
    int tlx = max(0, bb.tl().x), tly = max(0, bb.tl().y);
    int brx = min(nextImg.cols, bb.br().x), bry = min(nextImg.rows, bb.br().y);
    Rect retBB(tlx, tly, brx - tlx, bry - tly);
    
    return retBB;
}

void TLD::track(Rect &bbTrack, vector<Rect> &bbDetect)
{
    ///// debug
    bbTrack = BB_ERROR;
    bbDetect.clear();
    /////
    
    tracker = new MedianFlow(prevImg, nextImg);
    
    //track
    int trackerStatus;
    Rect trackerRet = tracker->trackBox(bb, trackerStatus);
    Rect trackerRetInside = getInside(trackerRet);
    
    //detect
    TYPE_DETECTOR_RET detectorRet;
    detector.dectect(nextImg, detectorRet);
    
    //integrate
    float trackSc = -1;
    Rect finalBB, finalBBInside;
    
    ////// just test
    //if(trackerStatus == MF_TRACK_SUCCESS && detector.calcSc(nextImg(_rect)) < 0.4)
    //{
    //    trackerStatus = !trackerStatus;
    //}
    //////
    
    if(trackerStatus != MF_TRACK_SUCCESS && detectorRet.size() == 0)
    {
        cerr << "Not visible." << endl;
        bb = BB_ERROR;
        
        delete tracker;
        return;
    }
    
    if(trackerStatus == MF_TRACK_SUCCESS)
    {
        float Sc = detector.calcSc(nextImg(trackerRetInside));
        cerr << "Track bb : " << trackerRet << " Sc : " << Sc << " Sr : " << detector.calcSr(nextImg(trackerRetInside)) << endl;

        trackSc = Sc;
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
            
            Q.push(i);
            belong[i] = cntBelong;
            
            while(!Q.empty())
            {
                int x = Q.front();
                Q.pop();
                belong[x] = cntBelong;
                for(auto &x : edges[x])
                {
                    if(belong[x] == -1) Q.push(x);
                }
            }
            
            cntBelong++;
        }
        
        vector<pair<TYPE_DETECTOR_BB, float> > clusterBB;
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
                
                Sc += detector.calcSc(nextImg(detectorRet[j]));
                
                count++;
            }
            
            x /= count;
            y /= count;
            height /= count;
            width /= count;
            Sc /= count;
            
            Rect _rect = Rect(cvRound(x), cvRound(y), cvRound(width), cvRound(height));
            
            cerr << "Cluster " << i << " : " << _rect  << " Sc : " << Sc << endl;
            
            clusterBB.push_back(make_pair(_rect, Sc));
        }
        
        cout << "Found " << cntBelong << " clusters." << endl;
        
        if(trackerStatus == MF_TRACK_SUCCESS)
        {
            int confidentDetections = 0;
            int lastId = 0;
            
            for(int i = 0; i < cntBelong; i++)
            {
                if(overlap(clusterBB[i].first, trackerRet) < 0.5 && clusterBB[i].second > trackSc)
                {
                    confidentDetections++;
                    lastId = i;
                }
            }
            
            if(confidentDetections == 1)
            {
                cerr << "Found a better match.. Regard it as the final result.." << endl;
                finalBB = finalBBInside = clusterBB[lastId].first;
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
                
                cx = (10 * trackerRet.x + cx) / (10 + closeDetections);
                cy = (10 * trackerRet.y + cy) / (10 + closeDetections);
                cw = (10 * trackerRet.width + cw) / (10 + closeDetections);
                ch = (10 * trackerRet.height + ch) / (10 + closeDetections);
                
                finalBB = Rect(cvRound(cx), cvRound(cy), cvRound(cw), cvRound(ch));
                finalBBInside = getInside(finalBB);
            }
        }
        else
        {
            if(cntBelong == 1)
            {
                cerr << "Track failed but found only one cluster, regard it as the final result." << endl;
                finalBB = finalBBInside = clusterBB[0].first;
            }
            else
            {
                cerr << "Track failed and found too many clusters, discard detection result." << endl;
                cerr << "Not visible." << endl;
                bb = BB_ERROR;
                
                delete tracker;
                return;
            }
        }
    }
    
    cerr << "Final bb : " << finalBB << endl;
    
    if(trackerStatus == MF_TRACK_SUCCESS && detector.calcSc(nextImg(finalBBInside)) >= 0.5)
    {
        learner.learn(nextImg, finalBB);
    }
    else
    {
        cerr << "Not learning because change is too fast." << endl;
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