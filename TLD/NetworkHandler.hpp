//
//  NetworkHandler.hpp
//  TLD
//
//  Created by 陈裕昕 on 9/16/15.
//  Copyright (c) 2015 Fudan. All rights reserved.
//

#ifndef TLD_NetworkHandler_hpp
#define TLD_NetworkHandler_hpp

#include <string>
#include <curl/curl.h>

using namespace std;

namespace net {
    
    char ip[80];
    
    struct MemoryStruct {
        char *memory;
        size_t size;
    };
    
    static size_t
    WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp)
    {
        size_t realsize = size * nmemb;
        struct MemoryStruct *mem = (struct MemoryStruct *)userp;
        
        mem->memory = (char*)realloc(mem->memory, mem->size + realsize + 1);
        if(mem->memory == NULL) {
            /* out of memory! */
            printf("not enough memory (realloc returned NULL)\n");
            return 0;
        }
        
        memcpy(&(mem->memory[mem->size]), contents, realsize);
        mem->size += realsize;
        mem->memory[mem->size] = 0;
        
        return realsize;
    }
    
    bool post(const char *url, const std::string &data)
    {
        CURL *curl;
        CURLcode res;
        
        /* In windows, this will init the winsock stuff */
        curl_global_init(CURL_GLOBAL_ALL);
        
        /* get a curl handle */
        curl = curl_easy_init();
        if(curl) {
            /* First set the URL that is about to receive our POST. This URL can
             just as well be a https:// URL if that is what should receive the
             data. */
            curl_easy_setopt(curl, CURLOPT_URL, url);
            /* Now specify the POST data */
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
            
            /* Perform the request, res will get the return code */
            res = curl_easy_perform(curl);
            /* Check for errors */
            if(res != CURLE_OK)
                fprintf(stderr, "curl_easy_perform() failed: %s\n",
                        curl_easy_strerror(res));
            
            /* always cleanup */
            curl_easy_cleanup(curl);
        }
        curl_global_cleanup();
        
        return res == CURLE_OK;
    }
    
    bool get(const char *url, std::string &str)
    {
        CURL *curl_handle;
        CURLcode res;
        
        struct MemoryStruct chunk;
        
        chunk.memory = (char*)malloc(1);  /* will be grown as needed by the realloc above */
        chunk.size = 0;    /* no data at this point */
        
        curl_global_init(CURL_GLOBAL_ALL);
        
        /* init the curl session */
        curl_handle = curl_easy_init();
        
        /* specify URL to get */
        curl_easy_setopt(curl_handle, CURLOPT_URL, url);
        
        /* send all data to this function  */
        curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
        
        /* we pass our 'chunk' struct to the callback function */
        curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, (void *)&chunk);
        
        /* some servers don't like requests that are made without a user-agent
         field, so we provide one */
        curl_easy_setopt(curl_handle, CURLOPT_USERAGENT, "libcurl-agent/1.0");
        
        /* get it! */
        res = curl_easy_perform(curl_handle);
        
        /* check for errors */
        if(res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n",
                    curl_easy_strerror(res));
        }
        else {
            /*
             * Now, our chunk.memory points to a memory block that is chunk.size
             * bytes big and contains the remote file.
             *
             * Do something nice with it!
             */
            
            printf("%lu bytes retrieved\n", (long)chunk.size);
        }
        
        /* cleanup curl stuff */
        curl_easy_cleanup(curl_handle);
        
        str = chunk.memory;
        
        free(chunk.memory);
        
        curl_easy_getinfo(curl_handle, CURLINFO_LOCAL_IP, &ip);
        
        /* we're done with libcurl, so clean it up */ 
        curl_global_cleanup();
        
        return res == CURLE_OK;
    }
}

#endif
