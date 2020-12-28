#!/bin/bash
sudo rm -r "/usr/local/cuda"
sudo ln -s "/usr/local/cuda-$1" "/usr/local/cuda"
