//
//  systemStruct.h
//  MedianFlow
//
//  Created by 陈裕昕 on 14/11/8.
//  Copyright (c) 2014年 陈裕昕. All rights reserved.
//

#ifndef MedianFlow_systemStruct_h
#define MedianFlow_systemStruct_h

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

typedef float TYPE_OF_COORD;
typedef TYPE_OF_COORD TYPE_MF_COORD;

typedef Point_<TYPE_OF_COORD> TYPE_OF_PT;
typedef TYPE_OF_PT TYPE_MF_PT;
typedef Rect_<TYPE_MF_COORD> TYPE_MF_BB;

typedef pair<Mat, bool> TYPE_TRAIN_DATA;
typedef vector<TYPE_TRAIN_DATA> TYPE_TRAIN_DATA_SET;

typedef pair<Point2f, Point2f> TYPE_FERN_LEAF; // save pixel comparision
typedef vector<vector<TYPE_FERN_LEAF> > TYPE_FERN_FERNS; // save all ferns
typedef vector<pair<int, int> > TYPE_FERN_PNCOUNTER; // save vote data of a fern

typedef Rect TYPE_DETECTOR_BB;
struct TYPE_DETECTOR_SCANBB
{
    TYPE_DETECTOR_BB first;   //bb
    float second; //overlap
    char status;
    
    TYPE_DETECTOR_SCANBB(const TYPE_DETECTOR_BB &_bb):first(_bb), second(-1), status(0){};
    TYPE_DETECTOR_SCANBB(const TYPE_DETECTOR_BB &_bb, const float _ol):first(_bb), second(_ol), status(0){};
};
typedef vector<TYPE_DETECTOR_SCANBB> TYPE_DETECTOR_SCANBBS;
typedef vector<TYPE_DETECTOR_BB> TYPE_DETECTOR_RET;

static const bool OF_USE_OPENCV = 1;

static const TYPE_OF_PT PT_ERROR = TYPE_OF_PT(-1, -1);
static const TYPE_MF_BB BB_ERROR = TYPE_MF_BB(PT_ERROR, PT_ERROR);

static const int MF_HALF_PATCH_SIZE = 4; // NNC patch size
static const int MF_NPTS = 10; // number of points in the patch(both vertical and horizontal)
static const int MF_ERROR_DIST = 20; // threshold of detecting confusing condition

static const int MF_TRACK_SUCCESS = 0;
static const int MF_TRACK_F_PTS = -1; // number of points after filtering is too little
static const int MF_TRACK_F_BOX = -2; // result box is out of bounds
static const int MF_TRACK_F_CONFUSION = -3; // tracking result is disordered
static const int MF_TRACK_F_BOX_SMALL = -4; // input box is too small

static const int MF_REJECT_OFERROR = 1 << 0; // filtered by OF error
static const int MF_REJECT_NCC = 1 << 1; // filtered by NCC
static const int MF_REJECT_FB = 1 << 2; // filtered by Forward-Backward

static const Scalar COLOR_GREEN = Scalar(156, 188, 26);
static const Scalar COLOR_BLUE = Scalar(219, 152, 52);
static const Scalar COLOR_BLACK = Scalar(94, 73, 52);
static const Scalar COLOR_WHITE = Scalar(241, 240, 236);
static const Scalar COLOR_YELLOW = Scalar(15, 196, 241);
static const Scalar COLOR_RED = Scalar(60, 76, 231);
static const Scalar COLOR_PURPLE = Scalar(182, 89, 155);

static const int LEARNER_N_WARPED = 10; // number of warped images to generate
static const int LEARNER_N_GOOD_BB = 10; // number of bounding boxes which are teated as good bounding boxes
                                         // and will be sent to update detector
static const float LEARNER_TH_OL = 0.2; // if overlap of a patch is small than 0.2 then it will be labeled as negative.

static const float VAR_FACTOR = 0.5;

static const float FERN_TH_POS = 0.5; // if avgP(1 | x) >= FERN_TH_POS then x will be classified as Positive
static const float FERN_TH_NEG = 1 - FERN_TH_POS;

static const int NN_MODEL_SIZE = 100; // size of POS model and NEG model
static const int NN_PATCH_SIZE = 15; // all input image will be resize to (NN_PATCH_SIZE * NN_PATCH_SIZE)
static const float NN_TH_POS = 0.65; // if Sr(patch) >= NN_TH_POS then  patch will be classified as Positive
static const float NN_MARGIN = 0.1; // if Sr(patch) - NN_TH_POS  < NN_MARGIN then it will be used for update
                                    // i.e. patch is not similar as samples in current model

static const bool CLASS_POS = true;
static const bool CLASS_NEG = false;

static const float DETECTOR_TH_OL = 0.2;

static const int DETECTOR_NFERNS = 10;
static const int DETECTOR_NLEAVES = 13;
static const int DETECTOR_MIN_BB_SIZE = 20; // minimum scanning bounding box size

static const float DETECTOR_WARP_NOISE = 5;
static const bool DETECTOR_WARP_BLUR = true;
static const float DETECTOR_WARP_SCALE = 0.02;
static const float DETECTOR_WARP_ANGLE = 10. / 180 * CV_PI;

// P-expert生成正样本时进行的warp参数
static const float DETECTOR_UPDATE_WARP_NOISE = 5;
static const bool DETECTOR_UPDATE_WARP_BLUR = true;
static const float DETECTOR_UPDATE_WARP_SCALE = 0.02;
static const float DETECTOR_UPDATE_WARP_ANGLE = 5. / 180 * CV_PI;

static const float DETECTOR_TH_GOOD_BB = 0.6;
static const float DETECTOR_TH_BAD_BB = 0.2;
static const int DETECTOR_N_GOOD_BB = 10;
static const int DETECTOR_N_WARPED = 10;

static const char DETECTOR_ACCEPTED = 1;
static const char DETECTOR_REJECT_VAR = 2;
static const char DETECTOR_REJECT_RF = 3;
static const char DETECTOR_REJECT_NN = 4;

static const bool NCC_USE_OPENCV = 0; // 1(lower speed): use matchTemplate(), 0(faster)

#endif
