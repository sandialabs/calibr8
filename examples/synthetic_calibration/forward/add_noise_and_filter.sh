#!/bin/bash

dmg_file="../../../source/calibr8/test/mesh/notch2D/notch2D.dmg"
smb_noiseless="notch2D_small_J2_plane_stress_synthetic"
smb_noisy="noisy"
smb_filtered="filtered"

num_steps=4

seed=22
noise_level="0.001"

poly_order=3
power_kernel_exponent=2
epsilon_multiplier=2.6

render ${dmg_file} "${smb_noiseless}/" "${smb_noiseless}_viz"

perturber ${dmg_file} "${smb_noiseless}/" ${num_steps} ${seed} ${noise_level} "${smb_noisy}/"
render ${dmg_file} "${smb_noisy}/" "${smb_noisy}_viz"

moving_least_squares ${dmg_file} "${smb_noisy}/" ${num_steps} ${poly_order} ${power_kernel_exponent} ${epsilon_multiplier} "${smb_filtered}/"
render ${dmg_file} "${smb_filtered}/" "${smb_filtered}_viz"
