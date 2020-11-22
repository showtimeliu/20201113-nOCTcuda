#include "OCTCUDAHeader.cuh"

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


__global__ void calculateMean(float* pfMatrix, float* pfMean, int nNumberLines, int nLineLength) {
    __shared__ float pfSum[1024];

    int nPoint = blockIdx.x * blockDim.x + threadIdx.x;
    float fSum = 0.0;
    int nLine;
    int nNumber = nNumberLines / blockDim.y;
    int nPosition = threadIdx.y * nNumber * nLineLength + nPoint;
    for (nLine = 0; nLine < nNumber; nLine++) {
        fSum += pfMatrix[nPosition];
        nPosition += nLineLength;
    }   // for (int nLine
    pfSum[threadIdx.x * blockDim.y + threadIdx.y] = fSum;

    __syncthreads();
    if (threadIdx.y == 0) {
        fSum = 0;
        nPosition = threadIdx.x * blockDim.y;
        for (nLine = 0; nLine < blockDim.y; nLine++) {
            fSum += pfSum[nPosition];
            nPosition++;
        }   // for (nLine
        pfMean[nPoint] = fSum / nNumberLines;
    }
}   // void calculateMean


//__global__ void subtractMean(float* pfMatrix, float* pfMean, int nNumberLines, int nLineLength) {
//    int nPoint = threadIdx.x + blockIdx.x * blockDim.x;
//    float fMean0 = pfMean[0 * nLineLength + nPoint];
//    float fMean1 = pfMean[1 * nLineLength + nPoint];
//    float* pfLine = pfMatrix + nPoint;
//    for (int nLine = 0; nLine < nNumberLines; nLine += 2) {
//        *(pfLine) -= fMean0;
//        pfLine += nLineLength;
//        *(pfLine) -= fMean1;
//        pfLine += nLineLength;
//    }   // for (int nLine
//}   // void subtractMean


__global__ void subtractMean(float* pfMatrix, float* pfMean, int nNumberLines, int nLineLength) {
    int nPoint = blockIdx.x * blockDim.x + threadIdx.x;
    float fMean = pfMean[nPoint];
    int nLine;
    int nNumber = nNumberLines / blockDim.y;
    int nPosition = threadIdx.y * nNumber * nLineLength + nPoint;
    for (nLine = 0; nLine < nNumber; nLine++) {
        pfMatrix[nPosition] -= fMean;
        nPosition += nLineLength;
    }   // for (int nLine
}   // void subtractMean


__global__ void calculateMask(float* pfMask, int nLength, int nStart, int nStop, int nRound) {
    int nPoint = blockIdx.x * blockDim.x + threadIdx.x;
    pfMask[nPoint] = 0.0;
    if (nPoint < nLength) {
        if (nPoint >= nStart - nRound)
            if (nPoint < nStart)
                pfMask[nPoint] = sin(0.5 * nPoint);
            else
                if (nPoint < nStop)
                    pfMask[nPoint] = 1.0;
                else
                    if (nPoint < nStop + nRound)
                        pfMask[nPoint] = sin(0.5 * nPoint);
    }   // if (nPoint
}   // void calculateMask


__global__ void applyMask(cufftComplex* pcMatrix, float* pfMask, int nNumberLines, int nLineLength) {
    int nPoint = blockIdx.x * blockDim.x + threadIdx.x;
    float fMask = pfMask[nPoint];
    int nLine;
    int nNumber = nNumberLines / blockDim.y;
    int nPosition = threadIdx.y * nNumber * nLineLength + nPoint;
    for (nLine = 0; nLine < nNumber; nLine++) {
        pcMatrix[nPosition].x *= fMask;
        pcMatrix[nPosition].y *= fMask;
        nPosition += nLineLength;
    }   // for (int nLine
}   // void subtractMean


__global__ void calculatePhase(cufftComplex* pcMatrix, float* pfPhase, int nNumberLines, int nLineLength) {
    int nPosition = (blockIdx.y * blockDim.y + threadIdx.y) * nLineLength + (blockIdx.x * blockDim.x + threadIdx.x);
    pfPhase[nPosition] = atan2(pcMatrix[nPosition].y, pcMatrix[nPosition].x);
}   // void calculatePhase