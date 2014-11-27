//
//  VarClassifier.cpp
//  TLD
//
//  Created by 陈裕昕 on 14/11/6.
//  Copyright (c) 2014年 Fudan. All rights reserved.
//

#include "VarClassifier.h"

VarClassifier::VarClassifier(const Mat &img)
{
    // CV_32F causes overflow ?
    integral(img, sum, sqsum, CV_64F);
}

float VarClassifier::getVar(const Rect &bb)
{
    // intermediate variable must be double ?
    int tlx = bb.tl().x, tly = bb.tl().y, brx = bb.br().x, bry = bb.br().y;
    double sumTl, sumTr, sumBl, sumBr;
    double sqsumTl, sqsumTr, sqsumBl, sqsumBr;
    
    sumTl = sum.at<double>(tly, tlx);
    sumTr = sum.at<double>(tly, brx);
    sumBl = sum.at<double>(bry, tlx);
    sumBr = sum.at<double>(bry, brx);

    sqsumTl = sqsum.at<double>(tly, tlx);
    sqsumTr = sqsum.at<double>(tly, brx);
    sqsumBl = sqsum.at<double>(bry, tlx);
    sqsumBr = sqsum.at<double>(bry, brx);
    
    double mean = (sumBr - sumTr - sumBl + sumTl) / bb.area();
    double sqmean = (sqsumBr - sqsumTr - sqsumBl + sqsumTl) / bb.area();

    return sqmean - mean * mean;
}

bool VarClassifier::getClass(TYPE_DETECTOR_SCANBB &bb, float patternVar)
{
    return (bb.var = getVar(bb)) >= patternVar * VAR_FACTOR;
}