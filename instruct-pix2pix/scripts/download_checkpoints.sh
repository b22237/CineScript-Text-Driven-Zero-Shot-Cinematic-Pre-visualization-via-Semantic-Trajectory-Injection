#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p /DATA/soham/preet/data/instruct-pix2pix/checkpoints

curl -L -C - \
     --retry 20 \
     --retry-delay 5 \
     --retry-max-time 0 \
     http://instruct-pix2pix.eecs.berkeley.edu/instruct-pix2pix-00-22000.ckpt \
     -o /DATA/soham/preet/data/instruct-pix2pix/checkpoints/instruct-pix2pix-00-22000.ckpt