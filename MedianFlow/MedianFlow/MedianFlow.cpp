  //
//  MedianFlow.cpp
//  MedianFlow
//
//  Created by 陈裕昕 on 10/29/14.
//  Copyright (c) 2014 陈裕昕. All rights reserved.
//

#include <cmath>
#include "MedianFlow.h"

MedianFlow::MedianFlow()
{

}

MedianFlow::MedianFlow(const Mat &prevImg, const Mat &nextImg)
{
    if(prevImg.channels() == 3)
        cvtColor(prevImg, this->prevImg, CV_BGR2GRAY);
    else
        prevImg.copyTo(this->prevImg);
    
    if(prevImg.channels() == 3)
        cvtColor(nextImg, this->nextImg, CV_BGR2GRAY);
    else
        nextImg.copyTo(this->nextImg);
    
    this->prevImg.convertTo(this->prevImg, CV_32F);
    this->nextImg.convertTo(this->nextImg, CV_32F);
    
    opticalFlow = new OpticalFlow(this->prevImg, this->nextImg, OpticalFlow::USEOPENCV);
    opticalFlowSwap = new OpticalFlow(this->nextImg, this->prevImg, OpticalFlow::USEOPENCV);
}

MedianFlow::MedianFlow(const Mat &prevImg, const Mat &nextImg, ViewController *_viewController)
{
    if(prevImg.channels() == 3)
        cvtColor(prevImg, this->prevImg, CV_BGR2GRAY);
    else
        prevImg.copyTo(this->prevImg);
    
    if(prevImg.channels() == 3)
        cvtColor(nextImg, this->nextImg, CV_BGR2GRAY);
    else
        nextImg.copyTo(this->nextImg);
    
    this->prevImg.convertTo(this->prevImg, CV_32F);
    this->nextImg.convertTo(this->nextImg, CV_32F);
    
    opticalFlow = new OpticalFlow(this->prevImg, this->nextImg, OpticalFlow::USEOPENCV);
    opticalFlowSwap = new OpticalFlow(this->nextImg, this->prevImg, OpticalFlow::USEOPENCV);
    
    viewController = _viewController;
}

MedianFlow::~MedianFlow()
{
    delete opticalFlow;
    opticalFlow = NULL;
    
    delete opticalFlowSwap;
    opticalFlowSwap = NULL;
}

vector<Point2f> MedianFlow::generatePts(const Rect_<float> &_box)
{
    Point2f tl(max(0.f, _box.tl().x), max(0.f, _box.tl().y));
    Point2f br(min((float)prevImg.cols, _box.br().x), min((float)prevImg.rows, _box.br().y));
    Rect_<float> box(tl, br);
    
    float stepX = (float)(box.width - 2 * halfPatchSize) / (nPts - 1);
    float stepY = (float)(box.height - 2 * halfPatchSize) / (nPts - 1);
    int x0 = box.x + halfPatchSize;
    int y0 = box.y + halfPatchSize;
    
    vector<Point2f> ret;
    
    for(int fx = 0; fx < nPts; fx++)
    {
        for(int fy = 0; fy < nPts; fy++)
        {
            ret.push_back(Point2f(x0 + fx * stepX, y0 + fy * stepY));
        }
    }
    
    return ret;
}

bool MedianFlow::compare(const pair<float, int> &a, const pair<float, int> &b)
// caution : prefix static can only be specified inside the class definition
{
    return a.first < b.first;
}

bool MedianFlow::isPointInside(const Point2f &pt, const float alpha)
{
    int width = prevImg.cols, height = prevImg.rows;
    return (pt.x >= 0 + alpha) && (pt.y >= 0 + alpha) && (pt.x <= width - alpha) && (pt.y <= height - alpha);
}

bool MedianFlow::isBoxUsable(const Rect_<float> &rect)
{
    bool insideBr = isPointInside(rect.br()), insideTl = isPointInside(rect.tl());
    int width = prevImg.cols, height = prevImg.rows;
    
    if(rect.br().x < 0 || rect.br().y < 0 || rect.tl().x > width || rect.tl().y > height)
        return false;
    
    Rect_<float> _rect(rect);
    
    if(!insideTl) _rect.tl() = Point2f(0, 0);
    if(!insideBr) _rect.br() = Point2f(width, height);
    
    if(_rect.width < nPts || _rect.height < nPts)
        return false;
    
    if(_rect.width > width || _rect.height > height) return false;
    
    return true;
}

void MedianFlow::filterOFError(const vector<Point2f> &pts, vector<int> &rejected)
{
    for(int i = 0; i < pts.size(); i++)
    {
        if(!isPointInside(pts[i])) rejected[i] |= REJECT_OFERROR;
    }
}

void MedianFlow::filterFB(const vector<Point2f> &initialPts, const vector<Point2f> &FBPts, vector<int> &rejected)
{
    int size = int(initialPts.size());
    vector<pair<float, int> > V;
    
    for(int i = 0; i < size; i++)
    {
        if(rejected[i] & REJECT_OFERROR) continue;
        
        float dist = norm(Mat(initialPts[i]), Mat(FBPts[i]));
        V.push_back(make_pair(dist, i));
    }
    
    sort(V.begin(), V.end(), compare);
    
    for(int i = (int)V.size() / 2; i < V.size(); i++)
    {
        rejected[V[i].second] |= REJECT_FB;
    }
}

double MedianFlow::calcNCC(const cv::Mat &img0, const cv::Mat &img1)
{
    Mat vImg0, vImg1; // convert image to 1 dimension vector
    
    vImg0 = img0.clone();
    vImg1 = img1.clone();
    
    vImg0.reshape(0, vImg0.cols * vImg0.rows);
    vImg1.reshape(0, vImg1.cols * vImg1.rows);
    
    Mat v01 = vImg0.t() * vImg1;
    
    float norm0, norm1;
    
    norm0 = norm(vImg0);
    norm1 = norm(vImg1);
    
    return abs(v01.at<float>(0)) / norm0 / norm1;
}

void MedianFlow::filterNCC(const vector<Point2f> &initialPts, const vector<Point2f> &FPts, vector<int> &rejected)
{
    int size = int(initialPts.size());
    vector<pair<float, int> > V;
    
    for(int i = 0; i < size; i++)
    {
        if(rejected[i] & REJECT_OFERROR) continue;
        
        if(!isPointInside(initialPts[i], halfPatchSize)) continue;
        if(!isPointInside(FPts[i], halfPatchSize)) continue;
        
        Point2d win(halfPatchSize, halfPatchSize);
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
        rejected[V[i].second] |= REJECT_NCC;
    }
}

Rect_<float> MedianFlow::calcRect(const Rect_<float> &rect, const vector<Point2f> &pts, const vector<Point2f> &FPts, const vector<int> &rejected, int &status)
{
    const int size = int(pts.size());
    
    vector<float> dxs, dys;
    
    for(int i = 0; i < size; i++)
    {
        if(rejected[i]) continue;
        
        dxs.push_back(FPts[i].x - pts[i].x);
        dys.push_back(FPts[i].y - pts[i].y);
    }
    
    if(dxs.size() <= 1)
    {
        status = MEDIANFLOW_TRACK_F_PTS;
        return Rect_<float>();
    }
    
    sort(dxs.begin(), dxs.end());
    sort(dys.begin(), dys.end());
    
    float dx = dxs[dxs.size() / 2];
    float dy = dys[dys.size() / 2];
    Point2f delta(dx, dy);
    
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
    
    Rect_<float> ret(delta + rect.tl(), delta + rect.br());
    
    Point2f center((float)(ret.tl().x + ret.br().x) / 2, (float)(ret.tl().y + ret.br().y) / 2);
    Point2f tl(center.x - ret.width / 2 * ratio, center.y - ret.height / 2 * ratio);
    Point2f br(center.x + ret.width / 2 * ratio, center.y + ret.height / 2 * ratio);
    
    ret = Rect_<float>(tl, br);
    
    //ret = Rect_<float>(tl, Size2f(rect.width * ratio, rect.height * ratio));
    
    cout << ret << endl;
    
    for(int i = 0; i < size; i++)
    {
        if(rejected[i]) continue;
        
        float dist = norm(Mat(pts[i]), Mat(FPts[i]));
        
        absDist.push_back(dist);
    }

    sort(absDist.begin(), absDist.end());
    
    int medianAbsDist = absDist[(int)absDist.size() / 2];
    
    for(auto &i : absDist)
        //caution : must add '&'
    {
        i = abs(i - medianAbsDist);
    }
    
    sort(absDist.begin(), absDist.end());
    if(absDist[(int)absDist.size() / 2] > errorDist)
    {
        status = MEDIANFLOW_TRACK_F_CONFUSION;
        return Rect_<float>();
    }
    
    if(!isBoxUsable(ret))
    {
        status = MEDIANFLOW_TRACK_F_BOX;
        return Rect_<float>();
    }
    
    status = MEDIANFLOW_TRACK_SUCCESS;
    return ret;
}

Rect_<float> MedianFlow::trackBox(const Rect_<float> &inputBox, int &status)
{
    // size of the inputBox is assumed to be larger than (10 + 4 * 2) * (10 + 4 * 2)
    //assert(inputBox.width >= 10 + 4 * 2 && inputBox.height >= 10 + 4 * 2);
    //if(inputBox.width >= 10 + 4 * 2 && inputBox.height >= 10 + 4 * 2) return inputBox;
    
    vector<Point2f> pts = generatePts(inputBox);
    
    vector<Point2f> retF, retFB;
    
    opticalFlow->trackPts(pts, retF);
    opticalFlowSwap->trackPts(retF, retFB);

    vector<int> rejected(nPts * nPts);
    
    filterOFError(retF, rejected);
    filterOFError(retFB, rejected);
    
    filterFB(pts, retFB, rejected);
    filterNCC(pts, retF, rejected);
    
    // debug
    vector<Point2f> rPts, rPts2, rPts3;
    for(int i = 0; i < pts.size(); i++)
    {
        if(rejected[i]) continue;
        rPts.push_back(pts[i]);
        rPts2.push_back(retF[i]);
        rPts3.push_back(retFB[i]);
    }
    //viewController->drawCircles(rPts, Scalar(255, 255, 0));
    //viewController->drawCircles(rPts2);
    //viewController->drawCircles(rPts3, Scalar(23, 45, 214));
    //viewController->drawLines(rPts, rPts2);
    
    //cout << "number of points after filtering:" << rPts.size() << endl;
    // end debug
    
    Rect_<float> ret;
    
    ret = calcRect(inputBox, pts, retF, rejected, status);
    
    if(status != MedianFlow::MEDIANFLOW_TRACK_SUCCESS)
    {
        cout << "tracking failed " << status << endl;
        
        return Rect_<float>(OFError, OFError);
    }
    
    status = MEDIANFLOW_TRACK_SUCCESS;
    return ret;
}