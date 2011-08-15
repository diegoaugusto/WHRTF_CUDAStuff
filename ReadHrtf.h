/*
 *  ReadHrtf.h
 *  WHRTF
 *
 *  Created by Diego Gomes on 25/03/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#if !defined(READHRTF_CLI)
#define READHRTF_CLI

#if defined(READHRTF_SER)
#define READHRTF_EXT
#else
#define READHRTF_EXT extern
#endif

/************************Funções Exportadas**********************/

READHRTF_EXT float** readHrtf(int elev, int azim, char ear, int minCoef, int maxCoef);

#undef READHRTF_EXT

#endif