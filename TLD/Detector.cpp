//
//  Detector.cpp
//  TLD
//
//  Created by 陈裕昕 on 11/5/14.
//  Copyright (c) 2014 Fudan. All rights reserved.
//

#include "Detector.h"


Detector::Detector(const Mat &img, const Rect &_patternBB):
    rFClassifier(RandomFernsClassifier(DETECTOR_NFERNS, DETECTOR_NLEAVES))
{
    // assert : img.type() == CV_8U
    
    patternBB = _patternBB;
    
    img(patternBB).copyTo(pattern);
    imgW = img.cols;
    imgH = img.rows;
    
    Scalar mean, dev;
    meanStdDev(pattern, mean, dev);

    patternVar = pow(dev.val[0], 2.);
    cerr << "Variance of pattern is " << patternVar << endl;
    
    patchGenerator = PatchGenerator(0, 0, DETECTOR_WARP_NOISE, DETECTOR_WARP_BLUR, 1. - DETECTOR_WARP_SCALE, 1. + DETECTOR_WARP_SCALE, -DETECTOR_WARP_ANGLE, DETECTOR_WARP_ANGLE, -DETECTOR_WARP_ANGLE, DETECTOR_WARP_ANGLE);
    
    train(img, patternBB);
}

Detector::~Detector()
{
    
}

void Detector::train(const Mat &img, const TYPE_DETECTOR_BB &patternBB)
{
    genScanBB();
    
    genPosData(img, trainDataSet);
    genNegData(img, trainDataSet);
    
    cerr << "Train Random Ferns Classifier." << endl;
    rFClassifier.train(trainDataSet);
    cerr << "Finished.." << endl;
    
    cerr << "Train Nearest Neighbor Classifier" << endl;
    nNClassifier.trainInit(trainDataSet);
    cerr << "Finished.." << endl;
}

void Detector::update()
{
    cerr << "Update Random Ferns Classifier." << endl;
    rFClassifier.train(trainDataSet);
    cerr << "Update Nearest Neighbor Classifier" << endl;
    nNClassifier.train(trainDataSet);
}

float Detector::overlap(const TYPE_DETECTOR_BB &bb1, const TYPE_DETECTOR_BB &bb2)
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

void Detector::sortByOverlap(const TYPE_DETECTOR_BB &bb, bool rand)
{
    for(auto &it : scanBBs)
    {
        it.second = overlap(bb, it.first);
    }
    
    sort(scanBBs.begin(), scanBBs.end(), scanBBCmp);
    
    if(rand)
    {
        auto it = scanBBs.begin();
    
        for(; it != scanBBs.end() && it->second > DETECTOR_TH_BAD_BB; it++) ;
    
        random_shuffle(it, scanBBs.end());
    }

}

void Detector::genScanBB()
{
    int minWH = min(patternBB.width, patternBB.height);
    int width = patternBB.width * ((float)DETECTOR_MIN_BB_SIZE / minWH);
    int height = patternBB.height * ((float)DETECTOR_MIN_BB_SIZE / minWH);
    
    for(; width <= imgW && height <= imgH; width *= 1.2, height *= 1.2)
    {
        int dx = width * 0.1, dy = height * 0.1;
        for(int x = 0; x + width <= imgW; x += dx)
        {
            for(int y = 0; y + height <= imgH; y += dy)
            {
                Rect bb(x, y, width, height);
                scanBBs.push_back(TYPE_DETECTOR_SCANBB(bb, -1));
            }
            
            Rect bb(x, imgH - height, width, height);
            scanBBs.push_back(TYPE_DETECTOR_SCANBB(bb, -1));
        }
        
        Rect bb(imgW - width, imgH - height, width, height);
        scanBBs.push_back(TYPE_DETECTOR_SCANBB(bb, -1));
    }
    
    sortByOverlap(patternBB, true);
}

void Detector::genWarped(const Mat &img, Mat &warped)
{
    patchGenerator(img, Point(img.cols / 2, img.rows / 2), warped, img.size(), theRNG());
}

void Detector::genPosData(const Mat &img, TYPE_TRAIN_DATA_SET &trainDataSet)
{
    int count = 0;
    for(int i = 0; i < DETECTOR_N_GOOD_BB && scanBBs[i].second >= DETECTOR_TH_GOOD_BB; i++)
    {
        for(int j = 0; j < DETECTOR_N_WARPED; j++)
        {
            Mat warped;
            genWarped(img(scanBBs[i].first), warped);

            TYPE_TRAIN_DATA trainData(make_pair(warped, CLASS_POS));
            trainDataSet.push_back(trainData);
            
            count++;
        }
    }
    
    cerr << "Generate " << count << " positive samples." << endl;
}

void Detector::genNegData(const Mat &img, TYPE_TRAIN_DATA_SET &trainDataSet)
{
    int count = 0;
    VarClassifier varClassifier(img);
    
    for(auto it = scanBBs.end() - 1; it >= scanBBs.begin() && (*it).second <= DETECTOR_TH_BAD_BB; it--)
    {
        if(varClassifier.getClass(it->first, patternVar) == CLASS_NEG) continue;
     
        TYPE_TRAIN_DATA trainData(make_pair(img(it->first), CLASS_NEG));
        //tTrainData trainData(make_pair(img(it->first).clone(), cNeg));
        trainDataSet.push_back(trainData);
        
        count++;
    }
    
    cerr << "Generate " << count << " negative samples." << endl;
}

void Detector::dectect(const Mat &img, TYPE_DETECTOR_RET &ret)
{
    cerr << "Start detecting." << endl;

    if(!ret.empty()) ret.clear();
    
    VarClassifier varClassifier(img);
    
    int count = 0;
    int acVar = 0, acRF = 0;
    for(auto &scanBB : scanBBs)
    {
        Rect &bb = scanBB.first;
        count++;
        
        if(varClassifier.getClass(bb, patternVar) == CLASS_POS)
        {
            acVar++;
            if(rFClassifier.getClass(img(bb)) == CLASS_POS)
            {
                acRF++;
                
                if(nNClassifier.getClass(img(bb)))
                {
                    ret.push_back(bb);

                    scanBB.status = DETECTOR_ACCEPTED;
                }
                else
                {
                    scanBB.status = DETECTOR_REJECT_NN;
                }
            }
            else
            {
                scanBB.status = DETECTOR_REJECT_RF;
            }
        }
        else
        {
            scanBB.status = DETECTOR_REJECT_VAR;
        }
    }
    
    cerr << "- After Variance Classifier : " << acVar << " bounding boxes." << endl;
    cerr << "- After Random Ferns Classifier: " << acRF << " bounding boxes." << endl;
    cerr << "- After Nearest Neighbor Classifier " << ret.size() << " bounding boxes." << endl;
    
    cerr << "Finish detecting." << endl;
}

float Detector::calcSr(const cv::Mat &img)
{
    return nNClassifier.calcSr(img);
}

float Detector::calcSc(const cv::Mat &img)
{
    return nNClassifier.calcSc(img);
}