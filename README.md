# Low-Latency and Secure Computation Offloading Assisted by Hybrid Relay-Reflecting Intelligent Surface

This repository contains the matlab numerical routines of the paper:

[1] K.-H. Ngo, N. T. Nguyen, T. Q. Dinh, T.-M. Hoang and M. Juntti, "Low-Latency and Secure Computation Offloading Assisted by Hybrid Relay-Reflecting Intelligent Surface," in *Proc. Int. Conf. Advanced Technologies for Communications (ATC)*, Hanoi, Vietnam, 2021, pp. 306-311.

Please, cite the aforementioned paper if you use this code.

## Content of the repository

This repository contains four `.m` files:

1. `single_user.m`

Minimization of computation offloading latency in a secure Mobile Edge Computing (MEC) system assisted with Hybrid Relayi-Reflecting Intelligent Surfaces (HRRIS). Four methods for the intelligent surface are considered:
  - Fixed HRRIS
  - Dynamic HRRIS
  - RIS with random phases
  - RIS with optimized phases according to the following paper

[2] T. Bai, C. Pan, Y. Deng, M. Elkashlan, A. Nallanathan, and L. Hanzo, "Latency minimization for intelligent reflecting surface aided mobile edge computing," *IEEE J. Sel. Topics Signal Process.*, vol. 38, no. 11, pp. 2666–2682, Nov. 2020.

We consider a single-user scenario:
         
                 |           Eavesdropper (EVE)
                 | <-----------> O
                 |     xEVE      ^
                 |               |    User Equipment (UE)
                 | <-------------+-------> O
                 |     xU        |         ^
                 |           yEVE|       yU|
                 |               |         |
                 |     xH        v         v
                 O <--------------------------------> O
            Edge Node (EN)                          HRRIS      

2. `gen_channel_HRRIS_SecureMEC.m`, `LoS_channel.m`: auxiliary functions to generate channel realizations.
3. `water_filling.m`: the Water Filling algorithm, written by G. Levin, May, 2003.
