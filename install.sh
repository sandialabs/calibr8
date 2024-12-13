#!/bin/bash

if [[ $# -gt 0 ]]; then
  flavor=$1
else
  arch=$(uname)
  if [ $arch == "Linux" ]; then
    hostname=$(hostname)
    host_type=${hostname:0:3}
    if [ $host_type == "cee" ] || [ $host_type == "ecw" ]; then
      flavor="cee-shared"
    else
      flavor="linux-shared"
    fi
  elif [ $arch == "Darwin" ]; then
    flavor="osx-shared"
  else
    echo "!!! $arch is not supported !!!"
    exit
  fi
fi

nproc=4
echo "building flavor ${flavor} with ${nproc} processors"

source env/$flavor.sh
source capp-setup.sh
capp load
capp build -j $nproc 2>&1 | tee capp_build_${flavor}.log
