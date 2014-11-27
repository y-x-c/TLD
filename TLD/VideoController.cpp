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

VideoController::VideoController(string &path, bool _imageMode):
    curr(0), frame(0), cameraMode(false), imageMode(_imageMode)
{
    if(!imageMode)
    {
        videoCapture = new VideoCapture(path);
    }
    else
    {
        this->path = path;
        string frameFilename(path + "framenum.txt");
        FILE *fin = fopen(frameFilename.c_str(), "r");
        fscanf(fin, "%d", &totalFrame);
        fclose(fin);
    }
    
    // check if the video file is opened
}

VideoController::VideoController(int camera):
    curr(0), frame(0), cameraMode(true)
{
    videoCapture = new VideoCapture(camera);
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
            string fullPath = path + filename + ".jpg";
            frames[curr] = imread(fullPath);
            return true;
        }
        return false;
    }
    else
    {
        return videoCapture -> read(frames[curr]);
    }
    
    return false;
}

Size VideoController::frameSize()
{
    if(imageMode)
    {
        return frames[curr].size();
    }
    else
    {
        return Size(videoCapture->get(CV_CAP_PROP_FRAME_WIDTH), videoCapture->get(CV_CAP_PROP_FRAME_HEIGHT));
    }
}

int VideoController::frameNumber()
{
    return frame;
}

void VideoController::jumpToFrameNum(int num)
{
    while(frameNumber() < num) readNextFrame();
}