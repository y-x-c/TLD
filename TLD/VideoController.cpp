//
//  VideoController.cpp
//  MedianFlow
//
//  Created by 陈裕昕 on 10/16/14.
//  Copyright (c) 2014 陈裕昕. All rights reserved.
//

#include "VideoController.h"

VideoController::~VideoController()
{
    if(!imageMode) delete videoCapture;
}

VideoController::VideoController(const string &path):
    curr(0), frame(0), cameraMode(false), imageMode(false)
{
    videoCapture = new VideoCapture(path);
    
    _frameSize = Size(videoCapture->get(CV_CAP_PROP_FRAME_WIDTH), videoCapture->get(CV_CAP_PROP_FRAME_HEIGHT));
    // check if the video file is opened
    
    int width = videoCapture->get(CV_CAP_PROP_FRAME_WIDTH);
    int height = videoCapture->get(CV_CAP_PROP_FRAME_HEIGHT);

    cerr << width << " " << height << endl;
    
    if(width > 480 || height > 480) {
        int maxwh = max(width, height);
        _frameSize = Size(width * (480.f / maxwh), height * (480.f / maxwh));
    } else {
        _frameSize = Size(width, height);
    }

}

VideoController::VideoController(const string &_path, const string &_append):
    curr(0), frame(0), cameraMode(false), imageMode(true), append(_append), path(_path)
{
    string frameFilename(path + "framenum.txt");
    FILE *fin = fopen(frameFilename.c_str(), "r");
    fscanf(fin, "%d", &totalFrame);
    fclose(fin);
    
    Mat tmp = imread(path + "00001" + append);
    _frameSize = tmp.size();
}

VideoController::VideoController(int camera):
    curr(0), frame(0), cameraMode(true), imageMode(false)
{
    videoCapture = new VideoCapture(camera);
    
    int width = videoCapture->get(CV_CAP_PROP_FRAME_WIDTH);
    int height = videoCapture->get(CV_CAP_PROP_FRAME_HEIGHT);
    _frameSize = Size(width * (480.f /width), height * (480.f / width));
}

int VideoController::getWidth() {
    return videoCapture->get(CV_CAP_PROP_FRAME_WIDTH);
}

int VideoController::getHeight() {
    return videoCapture->get(CV_CAP_PROP_FRAME_HEIGHT);
}

double VideoController::getCurrMsec() {
    return videoCapture->get(CV_CAP_PROP_POS_MSEC);
}

Mat VideoController::getCurrFrame()
{
    return frames[curr];
}

Mat VideoController::getPrevFrame()
{
    return frames[curr ^ 1];
}

bool VideoController::readNextFrame()
{
    curr ^= 1;
    frame++;
    
    if(imageMode)
    {
        if(frame <= totalFrame)
        {
            char filename[20];
            sprintf(filename, "%.5d", frame);
            string fullPath = path + filename + append;
            frames[curr] = imread(fullPath);
            return true;
        }
        return false;
    }
    else
    {
        bool f = videoCapture -> read(frames[curr]);
        
        if(f)
        {
            cerr << _frameSize;
            resize(frames[curr], frames[curr], _frameSize);
        }
        
        return f;
    }
    
    return false;
}

Size VideoController::frameSize()
{
    return _frameSize;
}

int VideoController::frameNumber()
{
    return frame;
}

void VideoController::jumpToFrameNum(int num)
{
//    while(frameNumber() < num) readNextFrame();
    videoCapture->set(CV_CAP_PROP_POS_FRAMES, num);
}

void VideoController::jumpToTime(double msec) {
    videoCapture->set(CV_CAP_PROP_POS_MSEC, msec);
}