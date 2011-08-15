/*
 *  whrtfConstants.h
 *  WHRTF
 *
 *  Created by Diego Gomes on 24/05/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

// Constants
const char* DB8 = "db8";
const int MIN_SAMPLE_NUMBER = 24;
const int MAX_SAMPLE_NUMBER = 174;
const int MAX_HRTF_SIZE = 512;
const int NUM_COEF_WITHOUT_BEGINNING = 488;	// 512-24 = 488
const int NUM_FILTROS = 4;
const char* WAVELET[] = {"db8", "db8", "db8", "db8"};


const double db8[2][8] = {
	{-0.0105, 0.0328, 0.0308, -0.1870, -0.0279, 0.6308, 0.7148, 0.2303},
	{-0.2303, 0.7148, -0.6308, -0.0279, 0.1870, 0.0308, -0.0328, -0.0105}};