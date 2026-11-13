#!/bin/sh
set -eu

sudo -n nvpmodel -m 0
sudo -n jetson_clocks
nvpmodel -q
sudo -n jetson_clocks --show
