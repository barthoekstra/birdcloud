#!/usr/bin/env bash
# Makefile
#

#Setting names of program

SOURCE = "KNMI_vol_h5_to_ODIM_h5.c"
BINARY = "KNMI_vol_h5_to_ODIM_h5.x"

#Setting compiler and options

COMPILE="gcc -Wall -O"

#Setting include and library directories

IPATH=""
LPATH=""

#Setting libraries

LIBRARIES="-lhdf5_hl -lhdf5 -lz -lm"

#Compiling of source

converter:
	$COMPILE $SOURCE -o $BINARY $IPATH $LPATH $LIBRARIES

# gcc -Wall -O KNMI_vol_h5_to_ODIM_h5.c -o KNMI_vol_h5_to_ODIM_h5 -lhdf5_hl -lhdf5 -lz -lm
