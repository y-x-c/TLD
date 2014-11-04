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

using namespace std;
using namespace cv;

Rect readRect()
{
    int width, height, x, y;
    scanf("[%d x %d from (%d, %d)]\n", &width, &height, &x, &y);
    
    return Rect(x, y, width, height);
}

void testRandomFernsClassifier()
{
    string filename("/Users/Orthocenter/Developments/MedianFlow/car.mpg");
    VideoController videoController(filename);
    ViewController viewController(&videoController);
    
    videoController.readNextFrame();
    
    RandomFernsClassifier randomFernsClassifier(10, 13);
    
    RandomFernsClassifier::tTrainDataSet trainDataSet;
    
    cerr << "Draw 20 positive bounding boxs" << endl;
    
    for(int i = 0; i < 20; i++)
    {
        cerr << "No. " << i << endl;
        
        //Rect rect = viewController.getRect();
        Rect rect = readRect();
        
        cout << rect << endl;
        
        RandomFernsClassifier::tTrainData trainData(make_pair(videoController.getCurrFrame()(rect), randomFernsClassifier.cPos));
        
        trainDataSet.push_back(trainData);
    }
    
    cerr << "Draw 20 negative bounding boxs" << endl;
    
    for(int i = 0; i < 20; i++)
    {
        cerr << "No. " << i << endl;
        
        //Rect rect = viewController.getRect();
        Rect rect = readRect();
        
        cout << rect << endl;
        
        RandomFernsClassifier::tTrainData trainData(make_pair(videoController.getCurrFrame()(rect), randomFernsClassifier.cNeg));
        
        trainDataSet.push_back(trainData);
    }

    cerr << "training..." << endl;
    randomFernsClassifier.train(trainDataSet);
    cerr << "finished..." << endl;
    
    for(;;)
    {
        Rect rect = viewController.getRect();
        
        cout << randomFernsClassifier.getClass(videoController.getCurrFrame()(rect)) << endl;
    }
}

int main()
{
    freopen("data.txt", "r", stdin);
    testRandomFernsClassifier();
    return 0;
}