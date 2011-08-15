/*
 *  MathUtil.h
 *  WHRTF
 *
 *  Created by Diego Gomes on 25/03/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#if !defined(MATHUTIL_CLI)
#define MATHUTIL_CLI

#if defined(MATHUTIL_SER)
#define MATHUTIL_EXT
#else
#define MATHUTIL_EXT extern
#endif

extern int hrtfFileSize;

/************************Funções Exportadas**********************/

MATHUTIL_EXT float norm(float* vec, int vecLength);
MATHUTIL_EXT float sumSquares(float* vec, int vecLength);
MATHUTIL_EXT float** sumSquaresColumns(float** vec, int vecLength);
MATHUTIL_EXT short* findIndexesGreaterThan(float* vec, int vecLength, float minValue);
MATHUTIL_EXT float* shift(float* vec, int vecLength, short delay, int maxLength);
MATHUTIL_EXT int max(int* vec, int vecLength);
MATHUTIL_EXT float* multiplyPolynomialCoefficients(float* vecA, const int vecASize, float* vecB, const int vecBSize);

#undef MATHUTIL_EXT

#endif