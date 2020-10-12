void computeSadOverBlockOpenCl(
        int minYL,
        int minXL,
        int minYR,
        int minXR,
        int width,
        int height,
        int imageWidth,
        global const unsigned char* leftImageData,
        global const unsigned char* rightImageData,
        private int* sum) {

    *sum = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            *sum += abs(
                    leftImageData[((y + minYL) * imageWidth) + (x + minXL)] -
                    rightImageData[((y + minYR) * imageWidth) + (x + minXR)]);
        }
    }
}

void computeDisparityForPixelOpenCl(
        int y, 
        int x,
        int imageWidth,
        int imageHeight,
        int blockSize,
        int leftScanSteps,
        int rightScanSteps,
        global const unsigned char* leftImageData,
        global const unsigned char* rightImageData,
        global float* output) {

    float disparityBuf[512];
    int maxBlockStep = (blockSize - 1) / 2;

    int templateLeftHalfWidth = min(x, maxBlockStep);
    int templateRightHalfWidth = min(imageWidth - x - 1, maxBlockStep);
    int templateTopHalfHeight = min(y, maxBlockStep);
    int templateBottomHalfHeight = min(imageHeight - y - 1, maxBlockStep);

    int templateWidth = templateLeftHalfWidth + templateRightHalfWidth + 1;
    int templateHeight = templateTopHalfHeight + templateBottomHalfHeight + 1;

    int leftMinY = y - templateTopHalfHeight;
    int leftMinX = x - templateLeftHalfWidth;

    int rightMinStartX = max(0, x - leftScanSteps - templateLeftHalfWidth);
    int rightMaxStartX = min(imageWidth - templateWidth, x + rightScanSteps - templateLeftHalfWidth);

    int numSteps = rightMaxStartX - rightMinStartX;

    int bestIndex = 0;
    int bestSadValue = 2147483646; // value of std::numeric_limits<int>::max() - 1
    int zeroDisparityIndex = x - rightMinStartX - templateLeftHalfWidth;

    for (int xx = rightMinStartX; xx <= rightMaxStartX; xx++) {
        int sad = 0;
        computeSadOverBlockOpenCl(
            leftMinY,
            leftMinX,
            leftMinY, // Ys are aligned for the two images
            xx,
            templateWidth,
            templateHeight,
            imageWidth,
            leftImageData, 
            rightImageData,
            &sad);

        disparityBuf[xx - rightMinStartX] = sad;

        if (sad < bestSadValue) {
            bestSadValue = sad;
            bestIndex = xx - rightMinStartX;
        }
    }

    float disparity = (float)(abs(bestIndex - zeroDisparityIndex));
    if ((bestIndex == 0)
        ||
        (bestIndex == numSteps)
        ||
        (bestSadValue == 0)) {
        *output = disparity;
    } else { 
        float c3 = disparityBuf[bestIndex+1];
        float c2 = disparityBuf[bestIndex];
        float c1 = disparityBuf[bestIndex-1];

        *output = disparity - (0.5 * ((c3 - c1) / (c1 - (2*c2) + c3)));
    }
}

__kernel 
void computeDisparityOpenClKernel(
        int height,
        int width,
        int blockSize,
        int leftScanSteps,
        int rightScanSteps,
        __global const unsigned char* leftImageData,
        __global const unsigned char* rightImageData,
        __global float* disparityData) {

    int index = get_global_id(0);
    int y = index / width;
    int x = index % width;

    computeDisparityForPixelOpenCl(
        y,
        x,
        width,
        height,
        blockSize,
        leftScanSteps,
        rightScanSteps,
        leftImageData,
        rightImageData,
        disparityData + index);
}

