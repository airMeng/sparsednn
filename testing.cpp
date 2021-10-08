
#include <cnpy.h>
#include "mkl.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <time.h>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <x86intrin.h>
// we are doing AC = AB * BC, reduce across the B dimension
// binding B to the x dimension, A to the y dimension and C to the z dimension

#define Tsy 1
#define Tsz (128 / 1)
#define ST 1
#define Fx 1
#define Fy (Tsz/Fx)

//#define 64 (64 / 1 / Tsy)

#define Usy (Tsy * Fy)
#define Gsy Usy

#define Gy 1
#define Block_size (Gy * Gsy)
#define X86 1
#define ARM 0
#include <pthread.h>
#include <cstdlib>


struct thread_data {
        const float * __restrict__ AB_val;
        const float * __restrict__ AB_bias;
        const float * __restrict__ BC;
        float * AC;
        int start;
        int end;
};

void * mm(void * threadarg)
{
        struct thread_data *my_data = (struct thread_data * ) threadarg;
        const float * __restrict__ AB_val = my_data->AB_val;
        const float * __restrict__ AB_bias = my_data->AB_bias;
        const float * __restrict__ BC = my_data->BC;
        float * AC = my_data->AC;
        int start = my_data->start;
        int end = my_data->end;

#if X86
    __m256 ACC[4];
	__m256 RC, val;
#elif ARM
    float32x4_t ACC[4];
    float32x4_t RC, val;
#endif
    __m256 zero = _mm256_setzero_ps();
   // #pragma omp parallel for schedule(static) private(ACC,RC,val,zero)

	for(int C_block = start; C_block < end; C_block ++){

	int C_offset = C_block * (128 / 1);
	
	


#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif



#if X86
    for(int j=0; j < 4; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < 4; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif




}
//pthread_exit(NULL);
}

