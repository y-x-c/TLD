  //
//  MedianFlow.cpp
//  MedianFlow
//
//  Created by 陈裕昕 on 10/29/14.
//  Copyright (c) 2014 陈裕昕. All rights reserved.
//

#include "MedianFlow.h"

MedianFlow::MedianFlow()
{

}

MedianFlow::MedianFlow(const Mat &prevImg, const Mat &nextImg, ViewController *_viewController)
{
    this->prevImg = prevImg;

    this->nextImg = nextImg;
    
    opticalFlow = new OpticalFlow(this->prevImg, this->nextImg);
    opticalFlowSwap = new OpticalFlow(this->nextImg, this->prevImg);
    
    viewController = _viewController;
}

MedianFlow::~MedianFlow()
{
    delete opticalFlow;
    opticalFlow = NULL;
    
    delete opticalFlowSwap;
    opticalFlowSwap = NULL;
}

void MedianFlow::generatePts(const TYPE_MF_BB &_box, vector<TYPE_MF_PT> &ret)
{
    TYPE_MF_PT tl(max(0.f, _box.tl().x), max(0.f, _box.tl().y));
    TYPE_MF_PT br(min((float)prevImg.cols, _box.br().x), min((float)prevImg.rows, _box.br().y));
    TYPE_MF_BB box(tl, br);
    
    float stepX = (float)(box.width - 2 * MF_HALF_PATCH_SIZE) / (MF_NPTS - 1);
    float stepY = (float)(box.height - 2 * MF_HALF_PATCH_SIZE) / (MF_NPTS - 1);
    int x0 = box.x + MF_HALF_PATCH_SIZE;
    int y0 = box.y + MF_HALF_PATCH_SIZE;
    
    if(!ret.empty()) ret.clear();
    
    for(int fx = 0; fx < MF_NPTS; fx++)
    {
        for(int fy = 0; fy < MF_NPTS; fy++)
        {
            ret.push_back(TYPE_MF_PT(x0 + fx * stepX, y0 + fy * stepY));
        }
    }
}

bool MedianFlow::compare(const pair<float, int> &a, const pair<float, int> &b)
// caution : prefix static can only be specified inside the class definition
{
    return a.first < b.first;
}

bool MedianFlow::isPointInside(const TYPE_MF_PT &pt, const TYPE_MF_COORD alpha)
{
    int width = prevImg.cols, height = prevImg.rows;
    return (pt.x >= 0 + alpha) && (pt.y >= 0 + alpha) && (pt.x <= width - alpha) && (pt.y <= height - alpha);
}

bool MedianFlow::isBoxUsable(const TYPE_MF_BB &rect)
{
    int width = prevImg.cols, height = prevImg.rows;
 
    // bounding box is too large
    if(rect.width > width || rect.height > height) return false;
    
    // intersection between rect and img is too small
    TYPE_MF_COORD tlx = max((TYPE_MF_COORD)rect.tl().x, (TYPE_MF_COORD)0);
    TYPE_MF_COORD tly = max((TYPE_MF_COORD)rect.tl().y, (TYPE_MF_COORD)0);
    TYPE_MF_COORD brx = min((TYPE_MF_COORD)rect.br().x, (TYPE_MF_COORD)width);
    TYPE_MF_COORD bry = min((TYPE_MF_COORD)rect.br().y, (TYPE_MF_COORD)height);
    
    TYPE_MF_BB bb(tlx, tly, brx - tlx, bry - tly);
    if(bb.width <= 2 * MF_HALF_PATCH_SIZE || bb.height <= 2 * MF_HALF_PATCH_SIZE) return false;
 
    // otherwise
    return true;
}

void MedianFlow::filterOFError(const vector<TYPE_MF_PT> &pts, const vector<uchar> &status, vector<int> &rejected)
{
    for(int i = 0; i < pts.size(); i++)
    {
        if(status[i] == 0) rejected[i] |= MF_REJECT_OFERROR;
    }
}

void MedianFlow::filterFB(const vector<TYPE_MF_PT> &initialPts, const vector<TYPE_MF_PT> &FBPts, vector<int> &rejected)
{
    int size = int(initialPts.size());
    vector<pair<float, int> > V;
    
    for(int i = 0; i < size; i++)
    {
        if(rejected[i] & MF_REJECT_OFERROR) continue;
        
        float dist = norm(Mat(initialPts[i]), Mat(FBPts[i]));
        V.push_back(make_pair(dist, i));
    }
    
    sort(V.begin(), V.end(), compare);
    
    for(int i = (int)V.size() / 2; i < V.size(); i++)
    {
        rejected[V[i].second] |= MF_REJECT_FB;
    }
}

float MedianFlow::calcNCC(const cv::Mat &img0, const cv::Mat &img1)
{
    if(NCC_USE_OPENCV)
    {
        Mat nccMat;
        matchTemplate(img0, img1, nccMat, CV_TM_CCORR_NORMED);
        
        return nccMat.at<float>(0);
    }
    else
    {
        Mat v0, v1; // convert image to 1 dimension vector
        
        img0.convertTo(v0, CV_32F);
        img1.convertTo(v1, CV_32F);
        
        v0 = v0.reshape(0, v0.cols * v0.rows);
        v1 = v1.reshape(0, v1.cols * v1.rows);
        
        Scalar mean, stddev;
        meanStdDev(v0, mean, stddev);
        v0 -= mean.val[0];
        meanStdDev(v1, mean, stddev);
        v1 -= mean.val[0];
        
        Mat v01 = v0.t() * v1;
        
        float norm0, norm1;
        
        norm0 = norm(v0);
        norm1 = norm(v1);
        
        return v01.at<float>(0) / norm0 / norm1;
    }
}

void MedianFlow::filterNCC(const vector<TYPE_MF_PT> &initialPts, const vector<TYPE_MF_PT> &FPts, vector<int> &rejected)
{
    int size = int(initialPts.size());
    vector<pair<float, int> > V;
    
    for(int i = 0; i < size; i++)
    {
        if(rejected[i] & MF_REJECT_OFERROR) continue;
        
        if(!isPointInside(initialPts[i], MF_HALF_PATCH_SIZE)) continue;
        if(!isPointInside(FPts[i], MF_HALF_PATCH_SIZE)) continue;
        
        Point2d win(MF_HALF_PATCH_SIZE, MF_HALF_PATCH_SIZE);
        Point2d pt1(initialPts[i].x, initialPts[i].y);
        Point2d pt2(FPts[i].x, FPts[i].y);
        
        // must be int
        Rect_<int> rect0(pt1 - win, pt1 + win);
        Rect_<int> rect1(pt2 - win, pt2 + win);
        
        float ncc = calcNCC(this->prevImg(rect0), this->nextImg(rect1));
        
        V.push_back(make_pair(ncc, i));
    }
    
    sort(V.begin(), V.end(), compare);
    
    for(int i = int(V.size()) / 2; i < V.size(); i++)
    {
        rejected[V[i].second] |= MF_REJECT_NCC;
    }
}

TYPE_MF_BB MedianFlow::calcRect(const TYPE_MF_BB &rect, const vector<TYPE_MF_PT> &pts, const vector<TYPE_MF_PT> &FPts, const vector<TYPE_MF_PT> &FBPts, const vector<int> &rejected, int &status)
{
    const int size = int(pts.size());
    
    vector<TYPE_MF_COORD> dxs, dys;
    
    for(int i = 0; i < size; i++)
    {
        if(rejected[i]) continue;
        
        dxs.push_back(FPts[i].x - pts[i].x);
        dys.push_back(FPts[i].y - pts[i].y);
    }
    
    if(dxs.size() <= 1)
    {
        status = MF_TRACK_F_PTS;
        outputInfo("Tracker", "Error : Too little points after filter.");
        return BB_ERROR;
    }
    
    sort(dxs.begin(), dxs.end());
    sort(dys.begin(), dys.end());
    
    TYPE_MF_COORD dx = dxs[dxs.size() / 2];
    TYPE_MF_COORD dy = dys[dys.size() / 2];
    TYPE_MF_PT delta(dx, dy);
    
    vector<float> ratios;
    vector<float> absDist;
    
    for(int i = 0; i < size; i++)
    {
        if(rejected[i]) continue;
        
        for(int j = i + 1; j < size; j++)
        {
            if(rejected[j]) continue;
            
            float dist0 = norm(Mat(pts[i]), Mat(pts[j]));
            float dist1 = norm(Mat(FPts[i]), Mat(FPts[j]));
            float ratio = dist1 / dist0;
            
            ratios.push_back(ratio);
        }
    }
    
    sort(ratios.begin(), ratios.end());
    float ratio = ratios[ratios.size() / 2];
    
    TYPE_MF_BB ret(delta + rect.tl(), delta + rect.br());
    
    TYPE_MF_PT center((float)(ret.tl().x + ret.br().x) / 2, (float)(ret.tl().y + ret.br().y) / 2);
    TYPE_MF_PT tl(center.x - ret.width / 2 * ratio, center.y - ret.height / 2 * ratio);
    TYPE_MF_PT br(center.x + ret.width / 2 * ratio, center.y + ret.height / 2 * ratio);
    
    ret = TYPE_MF_BB(tl, br);
    
    for(int i = 0; i < size; i++)
    {
        if(rejected[i] == MF_REJECT_OFERROR) continue;
        
        float dist = norm(Mat(pts[i]), Mat(FBPts[i]));
        
        absDist.push_back(dist);
    }

    sort(absDist.begin(), absDist.end());
    
    float medianAbsDist = absDist[(int)absDist.size() / 2];
    
    for(auto &i : absDist) //caution : must add '&'
    {
        i = abs(i - medianAbsDist);
    }
    
    sort(absDist.begin(), absDist.end());

    if(medianAbsDist > MF_FB_ERROR_DIST)
    {
        status = MF_TRACK_F_CONFUSION;
        outputInfo("Tracker", "Error : Large foward-backward distance.");
        return BB_ERROR;
    }
    
    if(!isBoxUsable(ret))
    {
        status = MF_TRACK_F_BOX;
        outputInfo("Tracker", "Error : Result bounding box is unusable.");
        return BB_ERROR;
    }
    
    status = MF_TRACK_SUCCESS;
    outputInfo("Tracker", "Tracked successfully.");
    return ret;
}

TYPE_MF_BB MedianFlow::trackBox(const TYPE_MF_BB &inputBox, int &status)
{
    if(!isBoxUsable(inputBox))
    {
        status = MF_TRACK_F_BOX;
        outputInfo("Tracker", "Error : Input bounding box is unusable.");
        return BB_ERROR;
    }
    
    vector<TYPE_MF_PT> pts;
    generatePts(inputBox, pts);
    
    vector<TYPE_MF_PT> retF, retFB;
    vector<uchar> statusF, statusFB;
    
    retF = retFB = pts;
    
    opticalFlow->trackPts(pts, retF, statusF);
    opticalFlowSwap->trackPts(retF, retFB, statusFB);

    vector<int> rejected(MF_NPTS * MF_NPTS);
    
    filterOFError(retF, statusF, rejected);
    filterOFError(retFB, statusFB, rejected);
    
    filterFB(pts, retFB, rejected);
    filterNCC(pts, retF, rejected);
    
    TYPE_MF_BB ret;
    
    ret = calcRect(inputBox, pts, retF, retFB, rejected, status);
    
    if(status != MF_TRACK_SUCCESS)
    {
        return BB_ERROR;
    }
    
    return ret;
}