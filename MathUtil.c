/*
 *  MathUtil.c
 *  WHRTF
 *
 *  Created by Diego Gomes on 25/03/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Inclusão do respectivo módulo de definição */
#define MATHUTIL_SER
#include "MathUtil.h"
#undef MATHUTIL_SER

// ########################################
// prototypes
float norm(float* vec, int vecLength);
float sumSquares(float* vec, int vecLength);
float** sumSquaresColumns(float** vec, int vecLength);
short* findIndexesGreaterThan(float* vec, int vecLength, float minValue);
float* shift(float* vec, int vecLength, short delay, int maxLength);
int max(int* vec, int vecLength);
float* multiplyPolynomialCoefficients(float* vecA, const int vecASize, float* vecB, const int vecBSize);

/*
 Euclidian norm.
 */
float norm(float* vec, int vecLength) {
	float sum = sumSquares(vec, vecLength);
	return (float) sqrtf(sum);
}

float sumSquares(float* vec, int vecLength) {
	float sum = 0.0f;
	for (int i = 0; i < vecLength; i++) {
		sum = sum + (vec[i]*vec[i]);
	}
	return sum;
}

float** sumSquaresColumns(float** vec, int vecLength) {
	float** result = (float**) malloc(2*sizeof(float));
	float* y0 = (float*) malloc(150*sizeof(float));
	float* y1 = (float*) malloc(150*sizeof(float));
	
	for (int i = 0; i < vecLength; i++) {
		y0[i] = sumSquares(vec[0], i);
		y1[i] = sumSquares(vec[1], i);
	}
	
	result[0] = y0;
	result[1] = y1;
	
	return result;
}

short* findIndexesGreaterThan(float* vec, int vecLength, float minValue) {
	short* indexes = (short*)malloc(vecLength*sizeof(short));
	int j = 0;
	for (short i = 0; i < vecLength; i++) {
		if (vec[i] >= minValue) {
			indexes[j++] = i;
		}
	}
	return indexes;
}

/**
	@maxLength is the maximum lenght of the result array.
 */
float* shift(float* vec, int vecLength, short delay, int maxLength) {
	float* newVec = (float*) malloc(maxLength*sizeof(float));
	int j = 0;
	for (int i = delay; i < vecLength; i++) {
		newVec[j++] = vec[i];
	}
	return newVec;
}

int max(int* vec, int vecLength) {
	int max = 0;
	for (int k = 0; k < vecLength; k++) { 
		if (vec[k] > max) {
			max = vec[k];
		}
	}
	return max;
}

float* multiplyPolynomialCoefficients(float* vecA, const int vecASize, float* vecB, const int vecBSize) {
	int vecCSize = vecASize + vecBSize;
	float* vecC = (float*) malloc(vecCSize * sizeof(float));
	
	for (int i = 0; i < vecASize; i++) {
		for (int j = 0; j < vecBSize; j++) {
			vecC[i+j] = vecC[i+j] + (vecA[i] * vecB[j]);
		}
	}
	
	return vecC;
}