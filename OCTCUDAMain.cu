#include "OCTCUDAHeader.cuh"

using namespace std; 

int gnMode = -1;
int gnRawLineLength;
int gnRawNumberLines;
int gnCalibrationNumberLines;
int gnProcessNumberLines;
int gnProcessedNumberLines;
int gnPerpendicular;
int gnAllocationStatus = 0;
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

int main()
{
    cudaError_t cudaStatus;
    int nStatus;

    // get device count
    int m_nDeviceCount;
    nStatus = getDeviceCount(&m_nDeviceCount);
    if (nStatus == -1) {
        fprintf(stderr, "getDeviceCount failed!");
        return 1;
    }   // if (nStatus
    if (m_nDeviceCount == 1)
        fprintf(stdout, "%d device found.\n", m_nDeviceCount);
    else
        fprintf(stdout, "%d devices found.\n", m_nDeviceCount);
    fprintf(stdout, "\n");

    // loop through number of devices and get names
    char* m_strDeviceName;
    m_strDeviceName = (char*)malloc(256 * sizeof(char));
    for (int nDevice = 0; nDevice < m_nDeviceCount; nDevice++) {
        nStatus = getDeviceName(nDevice, m_strDeviceName);
        if (nStatus != 0) {
            fprintf(stderr, "can't get device name");
            return 1;
        }   // if (nStatus
        fprintf(stdout, "device %d : %s\n", nDevice, m_strDeviceName);
    }   // for (int nDevice
    free(m_strDeviceName);
    fprintf(stdout, "\n");

    // initialization
    initialize(1, 1024, 2048, 1024, 2048);

    // read data from binary file
    short* pnParallel;
    readPSSDOCTFile(&pnParallel);
    fprintf(stdout, "initialization complete\n");

    // change array type while copying into host memory array (need to do this in C#)
    std::copy(pnParallel, pnParallel + gnCalibrationNumberLines * gnRawLineLength, gpfRawCalibration);
    fprintf(stdout, "initialization complete\n");

    // start C++ clock
    auto t_start = std::chrono::high_resolution_clock::now();

    // start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaDeviceSynchronize();

    // pause
//    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    int nNumberReps = 1000;

    //// set up number of threads per block
    int nThreadsPerBlock;
    dim3 d3Threads;
    dim3 d3Blocks;

    for (int nRep = 0; nRep < nNumberReps; nRep++) {

        // loop through in chunks that can be processed
        int nAline, nNumberLinesInChunk;
        for (nAline = 0; nAline < gnCalibrationNumberLines; nAline += gnProcessNumberLines) {

            // copy chunk of data for processing
            nNumberLinesInChunk = nAline + gnProcessNumberLines - gnCalibrationNumberLines;
            if (nNumberLinesInChunk <= 0)
                nNumberLinesInChunk = gnProcessNumberLines;

            // copy every other line starting at 0
            gpuErrchk(cudaMemcpy2D(gpfProcessCalibration, gnProcessCalibrationPitch, gpfRawCalibration + (nAline + 0) * gnRawLineLength, 2 * gnProcessCalibrationPitch, gnProcessCalibrationPitch, nNumberLinesInChunk >> 1, cudaMemcpyHostToDevice));

            //float dSum;
            //float* pCurrent;
            //for (int nPoint = 0; nPoint < gnRawLineLength; nPoint++) {
            //    dSum = 0;
            //    pCurrent = gpfRawCalibration + nAline * gnRawLineLength + nPoint; 
            //    for (int nLine = 0; nLine < nNumberLinesInChunk >> 1; nLine++) {
            //        dSum += *(pCurrent);
            //        pCurrent += gnRawLineLength;
            //    }
            //    gpfReferenceEven[nPoint] = 2 * dSum / (nNumberLinesInChunk >> 1);
            //}

            // calculate reference
            d3Threads.x = 128;
            d3Threads.y = 1024 / d3Threads.x;
            d3Threads.z = 1;
            d3Blocks.x = gnProcessNumberLines / d3Threads.x;
            d3Blocks.y = 1;
            d3Blocks.z = 1;
            calculateMean<<<d3Blocks, d3Threads>>>(gpfProcessCalibration, gpfReferenceEven, nNumberLinesInChunk >> 1, gnRawLineLength);
            gpuErrchk(cudaPeekAtLastError());

            // subtract reference
            d3Threads.x = 32;
            d3Threads.y = 1024 / d3Threads.x;
            d3Threads.z = 1;
            d3Blocks.x = gnProcessNumberLines / d3Threads.x;
            d3Blocks.y = 1;
            d3Blocks.z = 1;
            subtractMean<<<d3Blocks, d3Threads>>>(gpfProcessCalibration, gpfReferenceEven, nNumberLinesInChunk >> 1, gnRawLineLength);
            gpuErrchk(cudaPeekAtLastError());

            // forward fft
            gpuErrchk(cudaMemset2D(gpcProcessDepthProfile, gnProcessDepthProfilePitch, 0.0, gnProcessDepthProfilePitch, gnProcessNumberLines >> 1));
            cufftExecR2C(gchForward, gpfProcessCalibration, gpcProcessDepthProfile);

            // calculate mask
            nThreadsPerBlock = 512;
            calculateMask<<<gnRawLineLength / nThreadsPerBlock, nThreadsPerBlock>>>(gpfCalibrationMask, gnRawLineLength, 50, 100, 16);

            // apply mask
            d3Threads.x = 32;
            d3Threads.y = 1024 / d3Threads.x;
            d3Threads.z = 1;
            d3Blocks.x = gnProcessNumberLines / d3Threads.x;
            d3Blocks.y = 1;
            d3Blocks.z = 1;
            applyMask<<<d3Blocks, d3Threads>>>(gpcProcessDepthProfile, gpfCalibrationMask, nNumberLinesInChunk >> 1, gnRawLineLength);
            gpuErrchk(cudaPeekAtLastError());

            // reverse fft
            cufftExecC2C(gchReverse, gpcProcessDepthProfile, gpcProcessSpectrum, CUFFT_INVERSE);

            // calculate phase
            d3Threads.x = 32;
            d3Threads.y = 1024 / d3Threads.x;
            d3Threads.z = 1;
            d3Blocks.x = gnRawLineLength / d3Threads.x;
            d3Blocks.y = (gnProcessNumberLines >> 1) / d3Threads.y;
            d3Blocks.z = 1;
            calculatePhase<<<d3Blocks, d3Threads>>>(gpcProcessSpectrum, gpfProcessPhase, nNumberLinesInChunk >> 1, gnRawLineLength);
            gpuErrchk(cudaPeekAtLastError());

        }   // for (nAline

        cudaDeviceSynchronize();
    }   // for (int nRep

    // stop C++ timer
    auto t_end = std::chrono::high_resolution_clock::now();

    // stop timer
    cudaEventRecord(stop);
    // can do transfers back to host memory here
    cudaEventSynchronize(stop);

    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    fprintf(stdout, "time in milliseconds = %f\n", elapsed_time_ms / nNumberReps);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    fprintf(stdout, "time in milliseconds = %f\n", milliseconds / nNumberReps);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    fprintf(stdout, "\n");

    // free memory
    cleanup();

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

