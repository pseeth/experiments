#!/usr/bin/env bash
autossh -M 6006 -f -nNT -L 6006:localhost:6006 cortex
autossh -M 8888 -f -nNT -L 8888:0.0.0.0:8888 cortex
