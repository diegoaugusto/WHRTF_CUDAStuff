/*
 *  MsgUtil.h
 *  WHRTF
 *
 *  Created by Diego Gomes on 25/03/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#if !defined(MSGUTIL_CLI)
#define MSGUTIL_CLI

#if defined(MSGUTIL_SER)
#define MSGUTIL_EXT
#else
#define MSGUTIL_EXT extern
#endif


/************************Funções Exportadas**********************/

MSGUTIL_EXT void showError(char* msg);

#undef MSGUTIL_EXT

#endif