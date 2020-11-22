#include "OCTCUDAHeader.cuh"

extern "C" {
    int gnMode;
    int gnRawLineLength;
    int gnRawNumberLines;
    int gnCalibrationNumberLines;
    int gnProcessNumberLines;
    int gnProcessedNumberLines;
    int gnPerpendicular;
    int gnAllocationStatus;
    int gnMidLength;

    float* gpfRawCalibration;
    float* gpfProcessCalibration;
    size_t gnProcessCalibrationPitch;

    // reference
    float* gpfReferenceEven;
    float* gpfReferenceOdd;

    // fft
    cufftComplex* gpcProcessDepthProfile;
    size_t gnProcessDepthProfilePitch;
    cufftHandle gchForward;

    // calibration mask
    int gnCalibrationStart;
    int gnCalibrationStop;
    int gnCalibrationRound;
    float* gpfCalibrationMask;

    // reverse fft
    cufftComplex* gpcProcessSpectrum;
    size_t gnProcessSpectrumPitch;
    cufftHandle gchReverse;

    // phase
    float* gpfProcessPhase;
    size_t gnProcessPhasePitch;

}


extern "C" {
    __declspec(dllexport) int getDeviceCount(int* nNumberDevices) {
        // check for GPU
        int nDevices = -1;
        int nRet = cudaGetDeviceCount(&nDevices);
        if (nRet == cudaSuccess) {
            *(nNumberDevices) = nDevices;
        }	// if (nRet
        return nRet;
    }	// int getDeviceCount
}	// extern "C"


extern "C" {
    __declspec(dllexport) int getDeviceName(int nDeviceNumber, char* strDeviceName) {
        // check for GPU
        cudaDeviceProp currentDevice;
        int nRet = cudaGetDeviceProperties(&currentDevice, nDeviceNumber);
        if (nRet == cudaSuccess) {
            sprintf(strDeviceName, "%s (%d SMs, %d b/s, %d t/b, %d t/s, %d shared kB, %d GB)",
                currentDevice.name,
                currentDevice.multiProcessorCount,
                currentDevice.maxBlocksPerMultiProcessor,
                currentDevice.maxThreadsPerBlock,
                currentDevice.maxThreadsPerMultiProcessor,
                currentDevice.sharedMemPerBlock / 1024,
                currentDevice.totalGlobalMem / 1024 / 1024 / 1024);

        }	// if (nRet
        return nRet;
    }	// int getDeviceName
}	// extern "C"


extern "C" {
    __declspec(dllexport) int cleanup() {
        // free memory allocations
        if (gnAllocationStatus == 1) {
            gpuErrchk(cudaFreeHost(gpfRawCalibration));
            gpuErrchk(cudaFree(gpfProcessCalibration));
            gpuErrchk(cudaFree(gpfReferenceEven));
            gpuErrchk(cudaFree(gpfReferenceOdd));
            gpuErrchk(cudaFree(gpcProcessDepthProfile));
            cufftDestroy(gchForward);
            gpuErrchk(cudaFree(gpfCalibrationMask));
            gpuErrchk(cudaFree(gpcProcessSpectrum));
            cufftDestroy(gchReverse);
            gpuErrchk(cudaFree(gpfProcessPhase));

            gnAllocationStatus = 0;
        }   // if (gnAllocationStatus
        return -1;
    }   // __declspec
}   // extern


extern "C" {
    __declspec(dllexport) int initialize(int nMode, int nRawLineLength, int nRawNumberLines, int nProcessNumberLines, int nProcessedNumberLines) {

        cleanup();

        // copy parameters to global parameters
        gnMode = nMode;
        gnRawLineLength = nRawLineLength;
        gnRawNumberLines = nRawNumberLines;
        gnProcessNumberLines = nProcessNumberLines;
        gnProcessedNumberLines = nProcessedNumberLines;

        // allocate memory
        gnPerpendicular = 0;
        switch (gnMode) {
        case 0: // SD-OCT
            gnPerpendicular = 0;
            gnCalibrationNumberLines = 1;
            break;
        case 1: // PS SD-OCT
            gnPerpendicular = 1;
            gnCalibrationNumberLines = gnRawNumberLines;
            break;
        case 2: // line field
            gnPerpendicular = 0;
            gnCalibrationNumberLines = 1;
            break;
        case 3: // OFDI
            gnPerpendicular = 0;
            gnCalibrationNumberLines = gnRawNumberLines;
            break;
        case 4: // PS OFDI
            gnPerpendicular = 1;
            gnCalibrationNumberLines = gnRawNumberLines;
            break;
        }   // switch (gnMode

        gpuErrchk(cudaMallocHost((void**)&gpfRawCalibration, (gnRawLineLength * gnCalibrationNumberLines) * sizeof(float)));
        gpuErrchk(cudaMallocPitch((void**)&gpfProcessCalibration, &gnProcessCalibrationPitch, gnRawLineLength * sizeof(float), gnProcessNumberLines >> 1));

        gpuErrchk(cudaMalloc((void**)&gpfReferenceEven, gnRawLineLength * sizeof(float)));
        gpuErrchk(cudaMalloc((void**)&gpfReferenceOdd, gnRawLineLength * sizeof(float)));

        gnMidLength = (int)(gnRawLineLength / 2 + 1);
        gpuErrchk(cudaMallocPitch((void**)&gpcProcessDepthProfile, &gnProcessDepthProfilePitch, gnRawLineLength * sizeof(cufftComplex), gnProcessNumberLines >> 1));
        int nRank = 1;
        int pn[] = { gnRawLineLength };
        int nIStride = 1, nOStride = 1;
        int nIDist = gnProcessCalibrationPitch / sizeof(float);
        int nODist = gnProcessDepthProfilePitch / sizeof(cufftComplex);
        int pnINEmbed[] = { 0 };
        int pnONEmbed[] = { 0 };
        int nBatch = gnProcessNumberLines >> 1;
        cufftPlanMany(&gchForward, nRank, pn, pnINEmbed, nIStride, nIDist, pnONEmbed, nOStride, nODist, CUFFT_R2C, nBatch);

        gpuErrchk(cudaMalloc((void**)&gpfCalibrationMask, gnRawLineLength * sizeof(float)));

        gpuErrchk(cudaMallocPitch((void**)&gpcProcessSpectrum, &gnProcessSpectrumPitch, gnRawLineLength * sizeof(cufftComplex), gnProcessNumberLines >> 1));
        nIDist = gnProcessDepthProfilePitch / sizeof(cufftComplex);
        nODist = gnProcessSpectrumPitch / sizeof(cufftComplex);
        cufftPlanMany(&gchReverse, nRank, pn, pnINEmbed, nIStride, nIDist, pnONEmbed, nOStride, nODist, CUFFT_C2C, nBatch);

        gpuErrchk(cudaMallocPitch((void**)&gpfProcessPhase, &gnProcessPhasePitch, gnRawLineLength * sizeof(float), gnProcessNumberLines >> 1));


        gnAllocationStatus = 1;

        return -1;
    }   // int initialize
}   // extern


extern "C" {
    __declspec(dllexport) int getDataSDOCT(void* pnIMAQParallel) {
        return -1;
    }   // int getData
}   // extern


int readPSSDOCTFile(short** pnBuffer) {
    // read data from file
    std::ifstream fRawBinary("pdH.bin");
    // get size of file and move back to beginning
    fRawBinary.seekg(0, std::ios_base::end);
    std::size_t nSize = fRawBinary.tellg();
    fRawBinary.seekg(0, std::ios_base::beg);
    // allocate space for the array
    *pnBuffer = (short*)malloc(nSize);
    // read data
    fRawBinary.read((char*)(*pnBuffer), nSize);
    // close file
    fRawBinary.close();

    return -1;
}


extern "C" {
    __declspec(dllexport) int getDataPSSDOCT(void* pnIMAQParallel, void* pnIMAQPerpendicular) {


        int nAline, nPoint, nLocation;

        // copy to host memory
        for (nAline = 0; nAline < gnCalibrationNumberLines; nAline++)
            for (nPoint = 0; nPoint < gnRawLineLength; nPoint++) {
                nLocation = nAline * gnRawLineLength + nPoint;
                //                gpfRawCalibrationParallel[nLocation] = (short)v[nLocation];
            }   // for (nPoint


        return -1;
    }   // int getData
}   // extern


extern "C" {
    __declspec(dllexport) int getDataLineField(void* pdDAQ, void* pnIMAQParallel) {
        return -1;
    }   // int getData
}   // extern


extern "C" {
    __declspec(dllexport) int getDataOFDI(void* pnAlazar) {
        return -1;
    }   // int getData
}   // extern


extern "C" {
    __declspec(dllexport) int getDataPSOFDI(void* pnAlazar) {
        return -1;
    }   // int getData
}   // extern


extern "C" {
    __declspec(dllexport) int processCalibration() {
        // loop through section by section

          // loop though line by line in section

          // process parallel phase ramp

          // process perpendicular phase ramp if needed
        return -1;
    }   // int getData
}   // extern




int calculatePhaseRamp(void* pnRawData, int nPeakLeft, int nPeakRight, int nPeakRound, int nAmplitudeThreshold, int nAmplitudeLeft, int nAmplitudeRight, void* pfDepthProfile, void* pfPeak, void* pdAmplitude, void* pdPhaseRamp) {
    // forward fft of raw data (1)

    // check/save peak mask parameters (calculate mask if necessary)

    // apply mask to get cut peak

    // inverse fft to get complex spectrum (2)

    // calculate amplitude and phase plot

    // get left and right bounds of amplitude plot

    // unwrap phase plot

    return -1;
}   // int calculatePhaseRamp




extern "C" {
    __declspec(dllexport) int sendData(int nMode) {
        return -1;
    }   // int sendData
}   // extern


//__global__ void calculateMean(float* pfMatrix, float* pfMean, int nNumberLines, int nLineLength) {
//    int nPoint = threadIdx.x + blockIdx.x * blockDim.x;
//    float fSum = 0.0;
//    int nOffset = nPoint;
//    for (int nLine = 0; nLine < nNumberLines; nLine ++) {
//        fSum += pfMatrix[nOffset];
//        nOffset += nLineLength;
//    }   // for (int nLine
//    pfMean[nPoint] = fSum / ((float)nNumberLines);
//}   // void calculateReference

