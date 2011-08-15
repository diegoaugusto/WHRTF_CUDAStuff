/*
 *  MsgUtil.c
 *  WHRTF
 *
 *  Created by Diego Gomes on 25/03/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Inclusão do respectivo módulo de definição */
#define MSGUTIL_SER
#include "MsgUtil.h"
#undef MSGUTIL_SER

// ########################################
// prototypes
void showError(char* msg);

/*
 This functions shows a message error in the default output
 */
void showError(char* msg) {
	char* errorHeader = (char*) malloc(255*sizeof(char));
	strcpy(errorHeader, "ERROR: ");
	strcat(errorHeader, msg);
	puts(errorHeader);
	free(errorHeader);
}

