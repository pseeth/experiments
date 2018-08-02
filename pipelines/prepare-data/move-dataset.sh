#!/usr/bin/env bash

if [ ! -d ../../data ]; then
    mv `pwd`/data `pwd`/../..
    ln -s `pwd`/../../data data
fi