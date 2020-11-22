#pragma once

#pragma warning( push )
#pragma warning( disable : 26451 )

#ifndef _OCT_CUDA_HEADER_
#define _OCT_CUDA_HEADER_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <cufft.h>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iostream>
#include <thread>
#include <chrono>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

extern "C" {    __declspec(dllexport) int getDeviceCount(int* nNumberDevices);  }
extern "C" {    __declspec(dllexport) int getDeviceName(int nDeviceNumber, char* strDeviceName);    }
extern "C" {    __declspec(dllexport) int cleanup();    }
extern "C" {    __declspec(dllexport) int initialize(int nMode, int nRawLineLength, int nRawNumberLines, int nProcessNumberLines, int nProcessedNumberLines);   }
extern "C" {    __declspec(dllexport) int getDataSDOCT(void* pnIMAQParallel);   }
extern "C" {    __declspec(dllexport) int getDataPSSDOCT(void* pnIMAQParallel, void* pnIMAQPerpendicular);  }
extern "C" {    __declspec(dllexport) int getDataLineField(void* pdDAQ, void* pnIMAQParallel);  }
extern "C" {    __declspec(dllexport) int getDataOFDI(void* pnAlazar);  }
extern "C" {    __declspec(dllexport) int getDataPSOFDI(void* pnAlazar);    }
extern "C" {    __declspec(dllexport) int processCalibration(); }
extern "C" {    __declspec(dllexport) int sendData(int nMode);  }


int calculatePhaseRamp(void* pnRawData, int nPeakLeft, int nPeakRight, int nPeakRound, int nAmplitudeThreshold, \
    int nAmplitudeLeft, int nAmplitudeRight, void* pfDepthProfile, void* pfPeak, void* pdAmplitude, void* pdPhaseRamp); 
int readPSSDOCTFile(short** pnBuffer); 

__global__ void calculateMean(float* pfMatrix, float* pfMean, int nNumberLines, int nLineLength); 
__global__ void subtractMean(float* pfMatrix, float* pfMean, int nNumberLines, int nLineLength); 
__global__ void calculateMask(float* pfMask, int nLength, int nStart, int nStop, int nRound);
__global__ void applyMask(cufftComplex* pcMatrix, float* pfMask, int nNumberLines, int nLineLength); 
__global__ void calculatePhase(cufftComplex* pcMatrix, float* pfPhase, int nNumberLines, int nLineLength);


#endif _OCT_CUDA_HEADER_ // #ifndef _OCT_CUDA_HEADER_