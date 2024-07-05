export CAPP_FLAVOR=linux-shared

# comment out the lines below if these modules are not available
# and make sure you have equivalents on your system
module purge

module load aue/cmake/3.27.7
module load aue/git/2.42.0
module load aue/gcc/11.4.0
module load aue/openmpi/4.1.6-gcc-11.4.0
