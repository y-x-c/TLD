//
//  Detector.cpp
//  TLD
//
//  Created by 陈裕昕 on 11/5/14.
//  Copyright (c) 2014 Fudan. All rights reserved.
//

#include "Detector.h"


void Detector::init(const Mat &img, const Mat &imgB, const Mat &img32F, const Rect &_patternBB)
{
    // assert : img.type() == CV_8U
    
    patternBB = _patternBB;
    
    imgW = img.cols;
    imgH = img.rows;
    
    genScanBB();
    sortByOverlap(patternBB, true);

    rFClassifier.init(DETECTOR_NFERNS, DETECTOR_NLEAVES, scales, imgB, patternBB.width, patternBB.height);
    
    Scalar mean, dev;
    meanStdDev(img(scanBBs[0]), mean, dev);

    patternVar = pow(dev.val[0], 2.);
    
    stringstream info;
    info << "Variance of initial patch is " << patternVar;
    outputInfo("Detector", info.str());
    
    patchGenerator = PatchGenerator(0, 0, DETECTOR_WARP_NOISE, DETECTOR_WARP_BLUR, 1. - DETECTOR_WARP_SCALE, 1. + DETECTOR_WARP_SCALE, -DETECTOR_WARP_ANGLE, DETECTOR_WARP_ANGLE, -DETECTOR_WARP_ANGLE, DETECTOR_WARP_ANGLE);
    
    initTrain(img, imgB, img32F, patternBB);
}

Detector::~Detector()
{
    
}

void Detector::initTrain(const Mat &img, const Mat &imgB, const Mat &img32F, const TYPE_BBOX &patternBB)
{
    genPosData(img, imgB, img32F, trainDataSetNN, trainDataSetRF);
    genNegData(img, imgB, img32F, trainDataSetNN, trainDataSetRF);

    if(RND_SHUFFLE_STD)
    {
        random_shuffle(trainDataSetRF.begin(), trainDataSetRF.end());
    }
    else
    {
        for(int i = 1; i < trainDataSetRF.size(); i++)
        {
            int r = (float)theRNG() * i;
            swap(trainDataSetRF[i], trainDataSetRF[r]);
        }
    }
    
    rFClassifier.train(trainDataSetRF);
    nNClassifier.train(trainDataSetNN);
    
    nNClassifier.showModel();
}

void Detector::update()
{
    rFClassifier.train(trainDataSetRF);
    nNClassifier.train(trainDataSetNN);
}

float Detector::overlap(const TYPE_BBOX &bb1, const TYPE_BBOX &bb2)
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

void Detector::sortByOverlap(const TYPE_BBOX &bb, bool rand)
{
    for(auto &it : scanBBs)
    {
        it.overlap = overlap(bb, it);
    }
    
    sort(scanBBs.begin(), scanBBs.end(), TYPE_DETECTOR_SCANBB::cmpOL);
    
    if(rand)
    {
        auto it = scanBBs.begin();
    
        for(; it != scanBBs.end() && it->overlap >= BADBB_OL; it++) ;
    
        if(RND_SHUFFLE_STD)
        {
            random_shuffle(it, scanBBs.end());
        }
        else
        {
            int shift = (int)(it - scanBBs.begin());
            int end = int(scanBBs.size() - shift);
            for(int i = 0; i <= end; i++)
            {
                swap(scanBBs[i + shift], scanBBs[(float)theRNG() * i + shift]);
            }
        }
    }
}

void Detector::genScanBB()
{
    float  s;
    scales.push_back(1.);
    s = DETECTOR_GRID_SCALE_FAC;
    for(int i = 0; i <= DETECTOR_GRID_SCALE_HALF_NUM ; i++, s *= DETECTOR_GRID_SCALE_FAC) scales.push_back(s);
    s = 1.f / DETECTOR_GRID_SCALE_FAC;
    for(int i = 0; i <= DETECTOR_GRID_SCALE_HALF_NUM ; i++, s /= DETECTOR_GRID_SCALE_FAC) scales.push_back(s);
    
    for(int i = 0 ; i < scales.size(); i++)
    {
        int width = round(scales[i] * patternBB.width);
        int height = round(scales[i] * patternBB.height);
        
        int minSide = min(width, height);
        
        if(width > imgW || height > imgH || min(width, height) < DETECTOR_MIN_BB_SIZE) continue;
        
        int dx = round(minSide * DETECTOR_GRID_STEP_FAC), dy = round(minSide * DETECTOR_GRID_STEP_FAC);
        
        for(int x = 1; x + width <= imgW; x += dx)
        {
            for(int y = 1; y + height <= imgH; y += dy)
            {
                Rect bb(x, y, width, height);
                scanBBs.push_back(TYPE_DETECTOR_SCANBB(bb));
                scanBBs[scanBBs.size() - 1].scaleId = i;
            }
            
            Rect bb(x, imgH - height, width, height);
            scanBBs.push_back(TYPE_DETECTOR_SCANBB(bb));
            scanBBs[scanBBs.size() - 1].scaleId = i;
        }
    }
    
    stringstream info;
    info << "Generated " << scanBBs.size() << " scanning grids." << endl;
    outputInfo("Detector", info.str());
}

void Detector::genPosData(const Mat &img, const Mat &imgB, const Mat &img32F, TYPE_TRAIN_DATA_SET &trainDataSetNN, TYPE_TRAIN_DATA_SET &trainDataSetRF)
{
    int count = 0;
    
    // NN - POS
    trainDataSetNN.push_back(make_pair(img32F(scanBBs[0]), CLASS_POS));
    
    // RF - POS
    int tlx = img.cols, tly = img.rows, brx = 0, bry = 0;
    
    for(int i = 0; i < DETECTOR_N_GOOD_BB && scanBBs[i].overlap >= GOODBB_OL; i++)
    {
        tlx = min(tlx, scanBBs[i].tl().x);
        tly = min(tly, scanBBs[i].tl().y);
        brx = max(brx, scanBBs[i].br().x);
        bry = max(bry, scanBBs[i].br().y);
    }
    
    Point tl(tlx, tly), br(brx, bry);
    Rect bbHull(tl, br);
    
    int cx, cy;
    cx = round((double)(tlx + brx) / 2);
    cy = round((double)(tly + bry) / 2);
    
    for(int j = 0; j < DETECTOR_N_WARPED; j++)
    {
        Mat warped;
        
        if(j != 0)
        {
            patchGenerator(imgB, Point(cx, cy), warped, bbHull.size(), theRNG());

            // for optimum in RF::getcode()
            Mat tmp(imgB.size() ,CV_8U, Scalar::all(0));
            warped.copyTo(tmp(Rect(0, 0, bbHull.size().width, bbHull.size().height)));
            warped = tmp;
        }
        else
        {
            warped = imgB(bbHull);
        }

        for(int i = 0; i < DETECTOR_N_GOOD_BB && scanBBs[i].overlap >= GOODBB_OL; i++)
        {
            Rect rect(scanBBs[i].tl() - tl, scanBBs[i].br() - tl);
            
            TYPE_TRAIN_DATA trainData(make_pair(warped(rect), CLASS_POS));
            trainDataSetRF.push_back(trainData);
            
            count++;
        }
    }
    
    stringstream info;
    info << "Generated 1 positive example for NN, and " << count << " positive example(s) for RF.";
    outputInfo("Detector", info.str());
}

void Detector::genNegData(const Mat &img, const Mat &imgB, const Mat &img32F, TYPE_TRAIN_DATA_SET &trainDataSetNN, TYPE_TRAIN_DATA_SET &trainDataSetRF)
{
    int countRF = 0, countRFT = 0;
    int countNN = 0, countNNT = 0;
    VarClassifier varClassifier(img);
    
    auto it = scanBBs.begin();
    for(; it != scanBBs.end() && it->overlap >= BADBB_OL; it++) ;
    
    for(; it < scanBBs.end(); it++)
    {
        TYPE_DETECTOR_SCANBB &bb = *it;
        if(varClassifier.getClass(bb, patternVar) == CLASS_NEG) continue;
     
        int c = ((int)theRNG() % 2) ? CLASS_TEST_NEG : CLASS_NEG;
        
        // NN - NEG
        if(countNN <= DETECTOR_N_INIT_NN_NEG && c == CLASS_NEG)
        {
            TYPE_TRAIN_DATA trainDataNN(make_pair(img32F(bb), c));
            trainDataSetNN.push_back(trainDataNN);
            countNN++;
        }
        
        if(countNNT <= DETECTOR_N_INIT_NNT_NEG && c == CLASS_TEST_NEG)
        {
            TYPE_TRAIN_DATA trainDataNN(make_pair(img32F(bb), c));
            trainDataSetNN.push_back(trainDataNN);
            countNNT++;
        }
        
        // RF - NEG
        TYPE_TRAIN_DATA trainDataRF(make_pair(imgB(bb), c));
        trainDataSetRF.push_back(trainDataRF);
        
        if(c == CLASS_NEG) countRF++;
        if(c == CLASS_TEST_NEG) countRFT++;
    }
    
    stringstream info;
    info << "Generated " << countNN << " negative example(s) for NN, and " << countRF << " negative example(s) for RF.";
    outputInfo("Detector", info.str());

    info.clear();
    info << "Generated " << countNNT << " negative example(s) for NN testing, and " << countRFT << " negative example(s) for RF testing.";
    outputInfo("Detector", info.str());
}

void Detector::dectect(const Mat &img, const Mat &imgB, const Mat &img32F, TYPE_DETECTOR_RET &ret)
{
    ret.clear();
    rfRet.clear();
    
    VarClassifier varClassifier(img);
    
    int count = 0;
    int acVar = 0, acRF = 0;
    for(auto &scanBB : scanBBs)
    {
        TYPE_DETECTOR_SCANBB &bb = scanBB;
        count++;
        
        if(varClassifier.getClass(bb, patternVar) == CLASS_POS)
        {
            acVar++;
            if(rFClassifier.getClass(imgB(bb), bb) == CLASS_POS)
            {
                acRF++;
                rfRet.push_back(bb);
            }
            else
            {
                bb.status = DETECTOR_REJECT_RF;
            }
        }
        else
        {
            bb.status = DETECTOR_REJECT_VAR;
        }
    }
    
    if(rfRet.size() > DETECTOR_MAX_ENTER_NN)
    {
        sort(rfRet.begin(), rfRet.end(), TYPE_DETECTOR_SCANBB::cmpP);
        rfRet.resize(DETECTOR_MAX_ENTER_NN);
    }

    for(auto &bb : rfRet)
    {
        if(nNClassifier.getClass(img32F(bb), bb))
        {
            ret.push_back(bb);
            
            bb.status = DETECTOR_ACCEPTED;
        }
        else
        {
            bb.status = DETECTOR_REJECT_NN;
        }
    }
    
    stringstream info;
    info << "After Variance Classifier remains " << acVar << " bounding boxes.";
    outputInfo("Detector", info.str());
    
    stringstream info2;
    info2 << "After Random Ferns Classifier remains " << acRF << " bounding boxes.";
    outputInfo("Detector", info2.str());
    
    stringstream info3;
    info3 << "After Nearest Neighbor Classifier remains " << ret.size() << " bounding boxes.";
    outputInfo("Detector", info3.str());
}

void Detector::updataNNPara(const cv::Mat &img32F, TYPE_DETECTOR_SCANBB &sbb)
{
    nNClassifier.getClass(img32F(sbb), sbb);
}

float Detector::getNNThPos()
{
    return nNClassifier.thPos;
}
