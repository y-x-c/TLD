//
//  main.cpp
//  TLD
//
//  Created by 陈裕昕 on 11/4/14.
//  Copyright (c) 2014 Fudan. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "ViewController.h"
#include "VideoController.h"
#include "RandomFernsClassifier.h"
#include "NNClassifier.h"
#include "Detector.h"
#include "TLD.h"
#include "TLDSystemStruct.h"

using namespace std;
using namespace cv;

Rect readRect()
{
    int width, height, x, y;
    scanf("[%d x %d from (%d, %d)]\n", &width, &height, &x, &y);
    
    return Rect(x, y, width, height);
}

void testOnTLDDataset()
{
    string dir("/Users/Orthocenter/Developments/TLD/dataset2/04_pedestrian2/");

    string initFilename(dir + "init.txt");
    string retFilename(dir + "myRet6.txt");
    
    FILE *fin = fopen(initFilename.c_str(), "r");
    FILE *fout = fopen(retFilename.c_str(), "w");

    VideoController videoController(dir, true);
    ViewController viewController(&videoController);
    
    videoController.readNextFrame();

    int tlx, tly, brx, bry;
    fscanf(fin, "%d,%d,%d,%d", &tlx, &tly, &brx, &bry);
    fprintf(fout, "%d,%d,%d,%d\n", tlx, tly, brx, bry);
    
    Rect rect = Rect(Point2d(tlx, tly), Point2d(brx, bry));
    cerr << "Input Rect : " <<  rect << endl;
    
    viewController.refreshCache();
    viewController.drawRect(rect, COLOR_BLUE);
    viewController.showCache();
    waitKey();
    
    TLD tld(videoController.getCurrFrame(), rect);
    
    while(videoController.readNextFrame())
    {
        cerr << "Frame #" << videoController.frameNumber() << endl;
        tld.setNextFrame(videoController.getCurrFrame());
        
        Rect bbTrack;
        TYPE_DETECTOR_RET bbDetect;
        
        clock_t st = clock();
      
        tld.track();
        
        clock_t ed = clock();
        cerr << "Time : " << (double)(ed - st) / CLOCKS_PER_SEC * 1000 << "ms" << endl;

        viewController.refreshCache();
        viewController.drawRect(tld.getBB(), COLOR_GREEN, 2);
        viewController.showCache();
        
        cerr << endl;
        
        Rect retBB = tld.getBB();
        if(retBB == Rect(Point2d(-1, -1), Point2d(-1, -1)))
        {
            fprintf(fout, "NaN,NaN,NaN,NaN\n");
        }
        else
        {
            fprintf(fout, "%d,%d,%d,%d\n", retBB.tl().x, retBB.tl().y, retBB.br().x, retBB.br().y);
        }
    }
    
    fclose(fin);
    fclose(fout);
}

void testOnVideo()
{
    string filename("/Users/Orthocenter/Developments/TLD/3.m4v");
    
    VideoController videoController(filename);
    ViewController viewController(&videoController);
    
    videoController.readNextFrame();
    
    Rect rect = viewController.getRect();
    cerr << "Input Rect : " <<  rect << endl;
    
    viewController.refreshCache();
    viewController.drawRect(rect, COLOR_BLUE);
    viewController.showCache();
    waitKey();
    
    TLD tld(videoController.getCurrFrame(), rect);
    
    while(videoController.readNextFrame())
    {
        cerr << "Frame #" << videoController.frameNumber() << endl;
        tld.setNextFrame(videoController.getCurrFrame());
        
        Rect bbTrack;
        TYPE_DETECTOR_RET bbDetect;
        
        clock_t st = clock();
        
        tld.track();
        
        clock_t ed = clock();
        cerr << "Time : " << (double)(ed - st) / CLOCKS_PER_SEC * 1000 << "ms" << endl;
        
        viewController.refreshCache();
        viewController.drawRect(tld.getBB(), COLOR_GREEN, 2);
        viewController.showCache();
        
        cerr << endl;
    }

}

int main(int argc, char *argv[])
{
    testOnTLDDataset();
    //testOnVideo();
    return 0;
}