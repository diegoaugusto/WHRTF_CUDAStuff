/*
 *  sparseCoefficients.h
 *  WHRTF
 *
 *  Created by Diego Gomes on 27/05/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#if !defined(SPARSECOEF_CLI)
#define SPARSECOEF_CLI

#if defined(SPARSECOEF_SER)
#define SPARSECOEF_EXT
#else
#define SPARSECOEF_EXT extern
#endif

/************************Funções Exportadas**********************/

SPARSECOEF_EXT double** getSparseCoefficients(int elev, int azim, int ear, int* Gp_size);

#undef SPARSECOEF_EXT

#endif