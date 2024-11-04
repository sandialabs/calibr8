export FLAVOR=cee-shared
export NPROC=4

source env/$FLAVOR.sh
source capp-setup.sh
capp load
capp build -j $NPROC
