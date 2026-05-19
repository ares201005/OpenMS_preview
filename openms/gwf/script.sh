#!/bin/bash

PROGRAM=ga_2d_fermi_hubbard.py
METHOD=krylov
TOLERANCE=1e-4
SIZE=4

case $# in
    0)
        python3 "$PROGRAM" "$METHOD" "$TOLERANCE" "$SIZE"
        ;;
    1)
        python3 "$PROGRAM" "$METHOD" "$TOLERANCE" "$1"
        ;;
    2)
        python3 "$PROGRAM" "$METHOD" "$2" "$1"
        ;;
    3)
        python3 "$PROGRAM" "$3" "$2" "$1"
        ;;
    *)
        echo "Usage: $0 [size] [tolerance] [method]"
        exit 1
        ;;
esac
