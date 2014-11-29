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
    
    rFClassifier.init(DETECTOR_NFERNS, DETECTOR_NLEAVES, scales, patternBB.width, patternBB.height);
    
    Scalar mean, dev;
    meanStdDev(img(scanBBs[0]), mean, dev);

    patternVar = pow(dev.val[0], 2.);
    cerr << "Variance of pattern is " << patternVar << endl;
    
    patchGenerator = PatchGenerator(0, 0, DETECTOR_WARP_NOISE, DETECTOR_WARP_BLUR, 1. - DETECTOR_WARP_SCALE, 1. + DETECTOR_WARP_SCALE, -DETECTOR_WARP_ANGLE, DETECTOR_WARP_ANGLE, -DETECTOR_WARP_ANGLE, DETECTOR_WARP_ANGLE);
    
    updatePatchGenerator = PatchGenerator(0, 0, DETECTOR_UPDATE_WARP_NOISE, DETECTOR_UPDATE_WARP_BLUR, 1. - DETECTOR_UPDATE_WARP_SCALE, 1. + DETECTOR_UPDATE_WARP_SCALE, -DETECTOR_UPDATE_WARP_ANGLE, DETECTOR_UPDATE_WARP_ANGLE, -DETECTOR_UPDATE_WARP_ANGLE, DETECTOR_UPDATE_WARP_ANGLE);
    
    train(img, imgB, img32F, patternBB);
}

Detector::~Detector()
{
    
}

void Detector::train(const Mat &img, const Mat &imgB, const Mat &img32F, const TYPE_DETECTOR_BB &patternBB)
{
    genPosData(img, imgB, img32F, trainDataSetNN, trainDataSetRF);
    genNegData(img, imgB, img32F, trainDataSetNN, trainDataSetRF);
    
    for(int i = 1; i < trainDataSetRF.size(); i++)
    {
        int r = (float)theRNG() * i;
        swap(trainDataSetRF[i], trainDataSetRF[r]);
    }
    
    rFClassifier.train(trainDataSetRF);
    cerr << "Trained Random Ferns Classifier." << endl;
    
    nNClassifier.train(trainDataSetNN);
    cerr << "Trained Nearest Neighbor Classifier" << endl;
    
    nNClassifier.showModel();
}

void Detector::update()
{
    cerr << "Update Random Ferns Classifier." << endl;
    rFClassifier.train(trainDataSetRF);
    cerr << "Update Nearest Neighbor Classifier" << endl;
    nNClassifier.train(trainDataSetNN);
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
        it.overlap = overlap(bb, it);
    }
    
    sort(scanBBs.begin(), scanBBs.end(), TYPE_DETECTOR_SCANBB::cmpOL);
    
    if(rand)
    {
        auto it = scanBBs.begin();
    
        for(; it != scanBBs.end() && it->overlap >= DETECTOR_TH_BAD_BB; it++) ;
    
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
    //int minWH = min(patternBB.width, patternBB.height);
    //float widthf = patternBB.width * ((float)DETECTOR_MIN_BB_SIZE / minWH);
    //float heightf = patternBB.height * ((float)DETECTOR_MIN_BB_SIZE / minWH);
    
    //const float SCALES[] = {0.16151,0.19381,0.23257,0.27908,0.33490,0.40188,0.48225,
    //    0.57870,0.69444,0.83333,1,1.20000,1.44000,1.72800,
    //    2.07360,2.48832,2.98598,3.58318,4.29982,5.15978,6.19174};
    
    float fac = 1.2, s;
    scales.push_back(1.);
    s = fac;
    for(int i = 0; i <= 10 ; i++, s *= fac) scales.push_back(s);
    s = 1. / fac;
    for(int i = 0; i <= 10 ; i++, s /= fac) scales.push_back(s);
    
    //for(; widthf <= imgW && heightf <= imgH; widthf *= 1.125186016, heightf *= 1.125186016)
    for(int i = 0 ; i < scales.size(); i++)
    {
        int width = round(scales[i] * patternBB.width);
        int height = round(scales[i] * patternBB.height);
        //int width = cvRound(widthf);
        //int height = cvRound(heightf);
        
        int minSide = min(width, height);
        
        if(width > imgW || height > imgH || min(width, height) < DETECTOR_MIN_BB_SIZE) continue;
        
        int dx = round(minSide * 0.1), dy = round(minSide * 0.1);
        //for(int x = 0; x + width <= imgW; x += dx)
        for(int x = 1; x + width <= imgW; x += dx)
        {
            //for(int y = 0; y + height <= imgH; y += dy)
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
    
    cerr << "Genearte " << scanBBs.size() << " scan bounding boxes." << endl;
    
    sortByOverlap(patternBB, true);
}

//void Detector::genWarped(const Mat &img, Mat &warped)
//{
//    patchGenerator(img, Point(img.cols / 2, img.rows / 2), warped, img.size(), theRNG());
//}
//
//void Detector::genUpdateWarped(const cv::Mat &img, cv::Mat &warped)
//{
//    updatePatchGenerator(img, Point(img.cols / 2, img.rows / 2), warped, img.size(), theRNG());
//}

void Detector::genPosData(const Mat &img, const Mat &imgB, const Mat &img32F, TYPE_TRAIN_DATA_SET &trainDataSetNN, TYPE_TRAIN_DATA_SET &trainDataSetRF, const int nWarped)
{
    int count = 0;
    
    // NN - POS
    trainDataSetNN.push_back(make_pair(img32F(scanBBs[0]), CLASS_POS));
    
    // RF - POS
    int tlx = img.cols, tly = img.rows, brx = 0, bry = 0;
    
    for(int i = 0; i < DETECTOR_N_GOOD_BB && scanBBs[i].overlap >= DETECTOR_TH_GOOD_BB; i++)
    {
        tlx = min(tlx, scanBBs[i].tl().x);
        tly = min(tly, scanBBs[i].tl().y);
        brx = max(brx, scanBBs[i].br().x);
        bry = max(bry, scanBBs[i].br().y);
    }
    
    Point tl(tlx, tly), br(brx, bry);
    Rect bbHull(tl, br);
    
    int cx, cy;
    cx = cvRound((double)(tlx + brx) / 2);
    cy = cvRound((double)(tly + bry) / 2);
    
    for(int j = 0; j < nWarped; j++)
    {
        Mat warped;
        
        if(j != 0)
        {
            patchGenerator(imgB, Point(cx, cy), warped, bbHull.size(), theRNG());
        }
        else
        {
            warped = imgB(bbHull);
        }
        
        for(int i = 0; i < DETECTOR_N_GOOD_BB && scanBBs[i].overlap >= DETECTOR_TH_GOOD_BB; i++)
        {
            Rect rect(scanBBs[i].tl() - tl, scanBBs[i].br() - tl);
            
            TYPE_TRAIN_DATA trainData(make_pair(warped(rect), CLASS_POS));
            trainDataSetRF.push_back(trainData);
            
            count++;
        }
    }
    
    cerr << "Generate 1 NN positive samples and " << count << " RF positive samples." << endl;
}

void Detector::genNegData(const Mat &img, const Mat &imgB, const Mat &img32F, TYPE_TRAIN_DATA_SET &trainDataSetNN, TYPE_TRAIN_DATA_SET &trainDataSetRF)
{
    int countRF = 0, countRFT = 0;
    int countNN = 0, countNNT = 0;
    VarClassifier varClassifier(img);
    
    auto it = scanBBs.begin();
    for(; it != scanBBs.end() && it->overlap >= DETECTOR_TH_BAD_BB; it++) ;
    
    for(; it < scanBBs.end(); it++)
    {
        TYPE_DETECTOR_SCANBB &bb = *it;
        if(varClassifier.getClass(bb, patternVar) == CLASS_NEG) continue;
     
        int c = ((int)theRNG() % 2) ? CLASS_TEST_NEG : CLASS_NEG;
        
        // NN - NEG
        if(countNN <= 100 && c == CLASS_NEG)
        {
            TYPE_TRAIN_DATA trainDataNN(make_pair(img32F(bb), c));
            trainDataSetNN.push_back(trainDataNN);
            countNN++;
        }
        
        if(countNNT <= 100 && c == CLASS_TEST_NEG)
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
    
    cerr << "Generate " << countNN << " NN negative samples and " << countNNT << " NN negative test samples." << endl;
    cerr << "Generate " << countRF << " RF negative samples and " << countRFT << " RF negative test samples." << endl;
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
    
    if(rfRet.size() > 100)
    {
        sort(rfRet.begin(), rfRet.end(), TYPE_DETECTOR_SCANBB::cmpP);
        rfRet.resize(100);
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
    
    cerr << "- After Variance Classifier : " << acVar << " bounding boxes." << endl;
    cerr << "- After Random Ferns Classifier: " << acRF << " bounding boxes." << endl;
    cerr << "- After Nearest Neighbor Classifier " << ret.size() << " bounding boxes." << endl;
}

void Detector::updataNNPara(const cv::Mat &img32F, TYPE_DETECTOR_SCANBB &sbb)
{
    nNClassifier.getClass(img32F(sbb), sbb);
}

float Detector::getNNThPos()
{
    return nNClassifier.thPos;
}