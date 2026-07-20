## The Madgwick Filter
Implementation of the Madgwick filter in Rust. The Madgwick 
filter utilizes the quaternion representation of orientation
and gradient descent magic to overcome common problems like 
gimbal-lock and sensor drift. This gives you smooth and accurate
orientation tracking without an absolute reference to North.

## Repository
This repository contains an implementation of the Madgwick filter
in Rust, adapted from [Blake Johnson's implementation in C](https://github.com/bjohnsonfl/Madgwick_Filter). Along
with a minimal quaternion implementation and test program.

## Test Hardware
To test the filter I wired an MPU-6050 to a MicroBit:v2 and sent
the computed orientation over UART to a small Python program for 
rendering.

<video controls src="madquatter-1.mov" title="Title"></video>

## Notes
$
    \int    
$