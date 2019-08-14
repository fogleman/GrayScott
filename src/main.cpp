#define CL_SILENCE_DEPRECATION

// #if defined(__APPLE__) || defined(__MACOSX)
// #include <OpenCL/cl.hpp>
// #else
// #include <CL/cl.hpp>
// #endif

#include "cl.hpp"

#include <fstream>
#include <iostream>

const int w = 1024;
const int h = 1024;

const int iterations = 10000;

const int seedRate = 50;

const float feedRate = 0.025;
const float killRate = 0.055;

const float scale = 1;
const float diffusionRateA = 1 * scale;
const float diffusionRateB = 0.5 * scale;
const float timestep = 1 / scale;

const float centerWeight = -1;
const float adjacentWeight = 0.2;
const float diagonalWeight = 0.05;

const std::string kernelSource(R"(

__kernel void grayScott(
    global float *A0,
    global float *B0,
    global float *A1,
    global float *B1,
    const int w,
    const int h,
    const float centerWeight,
    const float adjacentWeight,
    const float diagonalWeight,
    const float feedRate,
    const float killRate,
    const float diffusionRateA,
    const float diffusionRateB,
    const float timestep,
    const int bufferIndex)
{
    // get pixel index
    const int i = get_global_id(0);

    // compute x and y coordinates of pixel
    const int x = i % w;
    const int y = i / w;

    // compute u and v coordinates of pixel in range [0, 1]
    const float u = (float)x / (w - 1);
    const float v = 1 - (float)y / (h - 1);

    // compute kill and feed rates for this pixel
    const float k = killRate + (u - 0.5) * 0;
    const float f = feedRate + (v - 0.5) * 0;

    // find neighboring pixels, wrapping around edges
    const int xp = x == 0 ? w - 1 : x - 1;
    const int xn = x == w - 1 ? 0 : x + 1;
    const int yp = y == 0 ? h - 1 : y - 1;
    const int yn = y == h - 1 ? 0 : y + 1;

    // figure out which buffers to use (kinda like double buffering)
    global float *A = bufferIndex == 0 ? A0 : A1;
    global float *B = bufferIndex == 0 ? B0 : B1;
    global float *newA = bufferIndex == 0 ? A1 : A0;
    global float *newB = bufferIndex == 0 ? B1 : B0;

    // get the values for A and B at this pixel
    const float a = A[i];
    const float b = B[i];

    // compute A diffusion
    float dda = 0;
    dda += a * centerWeight;
    dda += A[yp * w + xp] * diagonalWeight;
    dda += A[yp * w + xn] * diagonalWeight;
    dda += A[yn * w + xp] * diagonalWeight;
    dda += A[yn * w + xn] * diagonalWeight;
    dda += A[yp * w + x] * adjacentWeight;
    dda += A[yn * w + x] * adjacentWeight;
    dda += A[y * w + xp] * adjacentWeight;
    dda += A[y * w + xn] * adjacentWeight;

    // compute B diffusion
    float ddb = 0;
    ddb += b * centerWeight;
    ddb += B[yp * w + xp] * diagonalWeight;
    ddb += B[yp * w + xn] * diagonalWeight;
    ddb += B[yn * w + xp] * diagonalWeight;
    ddb += B[yn * w + xn] * diagonalWeight;
    ddb += B[yp * w + x] * adjacentWeight;
    ddb += B[yn * w + x] * adjacentWeight;
    ddb += B[y * w + xp] * adjacentWeight;
    ddb += B[y * w + xn] * adjacentWeight;

    // apply reaction diffusion formula
    const float da = diffusionRateA * dda - a * b * b + f * (1 - a);
    const float db = diffusionRateB * ddb + a * b * b - (f + k) * b;

    // write new A and B values for this pixel
    newA[i] = a + da * timestep;
    newB[i] = b + db * timestep;
}

)");

void savePPM(
    const std::string &path,
    const int width,
    const int height,
    const std::vector<float> &data)
{
    const float multiplier = 65535;
    std::ofstream out(path);
    out << "P3\n";
    out << width << " " << height << "\n";
    out << multiplier << "\n";
    int i = 0;
    const float lo = *std::min_element(data.begin(), data.end());
    const float hi = *std::max_element(data.begin(), data.end());
    std::cout << lo << ", " << hi << std::endl;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const float v = data[i++];
            float t = (v - lo) / (hi - lo);
            t = std::max(t, 0.f);
            t = std::min(t, 1.f);
            const int r = t * multiplier;
            out << r << " " << r << " " << r << "\n";
        }
    }
    out.close();
}

int main() {
    // get platform
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        return -1;
    }
    cl::Platform platform = platforms[0];
    std::cout << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

    // get device
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        return -1;
    }
    cl::Device device = devices[0];
    std::cout << device.getInfo<CL_DEVICE_NAME>() << std::endl;

    // compile program
    cl::Context context({device});
    cl::Program::Sources sources;
    sources.push_back({kernelSource.c_str(), kernelSource.size()});
    cl::Program program(context, sources);
    if (program.build({device}) != CL_SUCCESS) {
        std::cout
            << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
            << std::endl;
        return -1;
    }
    cl::Kernel kernel(program, "grayScott");

    // create & initialize cpu-side buffers
    std::vector<float> A(w * h, 1);
    std::vector<float> B(w * h, 0);
    srand(1);
    for (int i = 0; i < B.size(); i++) {
        if (rand() % seedRate == 0) {
            B[i] = 1;
        }
    }

    // make buffers
    const size_t numBytes = sizeof(A.front()) * A.size();
    cl::Buffer bufferA0(context, CL_MEM_READ_WRITE, numBytes);
    cl::Buffer bufferB0(context, CL_MEM_READ_WRITE, numBytes);
    cl::Buffer bufferA1(context, CL_MEM_READ_WRITE, numBytes);
    cl::Buffer bufferB1(context, CL_MEM_READ_WRITE, numBytes);

    // set arguments
    kernel.setArg(0, bufferA0);
    kernel.setArg(1, bufferB0);
    kernel.setArg(2, bufferA1);
    kernel.setArg(3, bufferB1);
    kernel.setArg(4, w);
    kernel.setArg(5, h);
    kernel.setArg(6, centerWeight);
    kernel.setArg(7, adjacentWeight);
    kernel.setArg(8, diagonalWeight);
    kernel.setArg(9, feedRate);
    kernel.setArg(10, killRate);
    kernel.setArg(11, diffusionRateA);
    kernel.setArg(12, diffusionRateB);
    kernel.setArg(13, timestep);

    // copy intial buffers over
    cl::CommandQueue queue(context, device);
    queue.enqueueWriteBuffer(bufferA0, CL_TRUE, 0, numBytes, A.data());
    queue.enqueueWriteBuffer(bufferB0, CL_TRUE, 0, numBytes, B.data());

    // run N iterations
    for (int i = 0; i < iterations; i++) {
        printf("\r  => %d of %d        \r", i+1, iterations);
        fflush(stdout);
        kernel.setArg(14, i % 2);
        queue.enqueueNDRangeKernel(
            kernel, cl::NullRange, cl::NDRange(A.size()), cl::NullRange);
        queue.finish();
    }
    printf("\n");

    // read out final buffers
    queue.enqueueReadBuffer(bufferA0, CL_TRUE, 0, numBytes, A.data());
    queue.enqueueReadBuffer(bufferB0, CL_TRUE, 0, numBytes, B.data());
    queue.finish();

    // write image
    savePPM("out.ppm", w, h, B);

    return 0;
}
