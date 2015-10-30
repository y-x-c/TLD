//
//  main.cpp
//  TLD
//
//  Created by 陈裕昕 on 11/4/14.
//  Copyright (c) 2014 Fudan. All rights reserved.
//
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "ViewController.h"
#include "VideoController.h"
#include "RandomFernsClassifier.h"
#include "NNClassifier.h"
#include "Detector.h"
#include "TLD.h"
#include "TLDSystemStruct.h"
#include "include/json.hpp"
#include "NetworkHandler.hpp"
#include <time.h>

using namespace std;
using namespace cv;

using json = nlohmann::json;

char FETCH_NEW_TASKS_URL[1000];
char GET_FILE_INFO_URL[1000];
char POST_RESULTS_URL[1000];
char UPDATE_STATE_URL[1000];

void loadURL(string configurePath){
    ifstream configFile(configurePath, std::ios::binary);
    
    string s = string(std::istreambuf_iterator<char>(configFile),
             std::istreambuf_iterator<char>());
    json config;
    config=json::parse(s.c_str());
    
    strcpy(FETCH_NEW_TASKS_URL, config["FETCH_NEW_TASKS_URL"].get<string>().c_str());
    strcpy(GET_FILE_INFO_URL, config["GET_FILE_INFO_URL"].get<string>().c_str());
    strcpy(POST_RESULTS_URL, config["POST_RESULTS_URL"].get<string>().c_str());
    strcpy(UPDATE_STATE_URL, config["UPDATE_STATE_URL"].get<string>().c_str());
    return;
}

void track(json task) {
    // get file path
    string fileInfo;
    string url = string(GET_FILE_INFO_URL) + "?movie_file_id=";
    url += task["movieId"].get<string>();
    string _res;
    while(!net::get(url.c_str(), _res)) {
        sleep(5);
    }
    
    json res = json::parse(_res);
    string filename = res["filePath"];
    
//    cerr << filename << endl;
    
    // track
    vector<json> results;
    
    VideoController videoController(filename);
//    ViewController viewController(&videoController);
    
    videoController.jumpToFrameNum(task["adFrame"]);
    videoController.readNextFrame();
    
    Point2i inTl(task["adX"], task["adY"]);
    Point2i inBr(task["adX"].get<int>() + task["adWidth"].get<int>(), task["adY"].get<int>() + task["adHeight"].get<int>());
    Rect rect(inTl, inBr);

//    cerr << "Input Rect : " <<  rect << endl;
    
//    viewController.refreshCache();
//    viewController.drawRect(rect, COLOR_BLUE);
//    viewController.showCache();
//    waitKey();
    
    TLD tld(videoController.getCurrFrame(), rect);
    
    int status = TLD_TRACK_SUCCESS;
    while(status == TLD_TRACK_SUCCESS && videoController.readNextFrame())
    {
        cerr << "Frame #" << videoController.frameNumber() << endl;
        tld.setNextFrame(videoController.getCurrFrame());
        
        Rect bbTrack;
        TYPE_DETECTOR_RET bbDetect;
        
        clock_t st = clock();
        
        status = tld.track();
        
        clock_t ed = clock();
        cerr << "Time : " << (double)(ed - st) / CLOCKS_PER_SEC * 1000 << "ms" << endl;
        
//        viewController.refreshCache();
//        viewController.drawRect(tld.getBB(), COLOR_GREEN, 2);
//        viewController.showCache();
        
        results.push_back({
            {"adX", tld.getBB().tl().x},
            {"adY", tld.getBB().tl().y},
            {"adWidth", tld.getBB().width},
            {"adHeight", tld.getBB().height},
            {"adFrame", videoController.frameNumber() - 1}
        });
        
        cerr << endl;
    }
    
//    cerr << results << endl;
//    cerr << "list=" + json(results).dump() << endl;
    
    // POST result
    
    //get datetime
    char datetime[80];
    time_t rawtime;
    struct tm * timeinfo;
    time (&rawtime);
    timeinfo = localtime (&rawtime);
    strftime (datetime, 80, "%F %R", timeinfo);
    
    url = string(POST_RESULTS_URL) + "?ad_info_id=" + task["adInfoId"].get<string>() + "&create_time=" + datetime + "&create_user_id=" + net::ip;
    
    while(!net::post(url.c_str(), "list=" + json(results).dump())) {
        sleep(5);
    }
}

void updateState(string ad_info_id, char state) {
    string data = "";
    string url = string(UPDATE_STATE_URL) + "?" + "ad_info_id=" + ad_info_id + "&" + "state=" + state;
//    cerr << url << " " << data << endl;

    while(!net::post(url.c_str(), data)) {
        sleep(5);
    }
}

void fetchNewTasks() {
    string _res;
    while(!net::get(FETCH_NEW_TASKS_URL, _res)) {
        sleep(5);
    }
    
    json res = json::parse(_res);
    
    for(auto task : res["list"]) {
        updateState(task["adInfoId"], '1');
        
        track(task);
        
        updateState(task["adInfoId"], '2');
    }
}

int main(int argc, char *argv[])
{
    //testOnTLDDataset();
    //testOnTLDDatasetAndOutputToFile();
//    testOnVideo();
//    testOnCamera();
    //trajectory();
    //stabilize();
    
    if(argc<=1){
        cerr<<"Usage: ./TLD {dir of configure.json}"<<endl;
        return -1;
    }
    
    
    loadURL(argv[1]); //Set up the URLs

    while(1) {
        fetchNewTasks();
        sleep(15);
    }
    return 0;
}


