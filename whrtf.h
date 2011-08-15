/*
 *  whrtf.h
 *  WHRTF
 *
 *  Created by Diego Gomes on 24/05/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#if !defined(WHRTF_CLI)
#define WHRTF_CLI

#if defined(WHRTF_SER)
#define WHRTF_EXT
#else
#define WHRTF_EXT extern
#endif

/************************Funções Exportadas**********************/

WHRTF_EXT float* whrtf (int elev, int azim, char ear, int* whrtfLength);
WHRTF_EXT float** getCoefSpars(int elev, int azim, char ear, int* G_size);
WHRTF_EXT float* getRespImp(int numFiltros, float** G, int* G_size, int* resultLength);

#undef WHRTF_EXT

#endif