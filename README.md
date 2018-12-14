Fast and Accurate Tensor Completion with Total Variation Regularized Tensor Trains (TTC) for Matlab&copy;/Octave&copy;
--------------------------------------------------------------------------------------------------

This package contains Matlab/Octave code for tensor completion using tensor trains with Total Variation Regularization (TTC).


1. Functions
------------

* demo

Demonstrates the usage of the TTC algorithm in tensor completions. 

* data_generate

Generate the missing entris and known entris for the data.

* b=contract(a)

Sums over all auxiliary indices of a Tensor Network a to return the underlying tensor b.

* y=dotkron(varargin)

Computes the row-wise right-Kronecker product of matrices A,B,C. Note that this computes the right-Kronecker product such that the order of the indices is maintained.

* cores=mpsvd(core,n,r)

Decompose a tensor into its tensor train form with TT-rank r.

* cores=mpsvd_op(core,n,r)

Decompose a tensor into its tensor train form with TT-rank r in an opposite direction.

* TN=tencom(y,u,r,init,varargin)

Tensor completion given the inputs, outputs, TT-ranks, and TT initialization with scalar outputs.

* [TN,e]=tencom_TV(y,u,r,init,Kn,idf,lambda,varargin)

Tensor completion given the inputs, outputs, TT-ranks, TT initialization, coordinate of known entries, dimensions identifier, and TV term parameter lambda with scalar outputs.

* TN=VecOtencom(y,u,r,init,varargin)

Tensor completion given the inputs, outputs, TT-ranks, and TT initialization with vector outputs.

* [TN,e]=VecOtencom_TV(y,u,r,init,Kn,idf,lambda,varargin)

Tensor completion given the inputs, outputs, TT-ranks, TT initialization,  coordinate of known entries, dimensions identifier, and TV term parameter lambda with vector outputs.

2. Reference
------------
"Fast and Accurate Tensor Completion with Total Variation Regularized Tensor Trains"

Authors: Ching-Yun Ko, Kim Batselier, Wenjian Yu, Ngai Wong


"Tensor Network alternating linear scheme for MIMO Volterra system identification"

Authors: Kim Batselier, Zhongming Chen, Ngai Wong
