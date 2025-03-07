---
title: "[EEE4610] Explore Sparse Attention Mechanisms"
date: 2025-03-03 10:07:00  # Created date
last_modified_at: 2025-03-04 10:10:00  # Modified date
---

Our team suggested some ideas, and our graduate assistant guided our research direction as follows:  
- Refer to some accelerator papers on existing sparsity operations and consider how to apply them to PIM.  
- Dynamically offload computational burdens from the GPU to PIM. 
This could be achieved by reviewing existing papers and proposing improvements.  

We must choose one of the following topics:  
1. Sparse Attention  
2. Dynamically Offloading from GPU to PIM  

First, let's take a look at the sparse attention methods first.
# Sparse Attention Mechanisms using PIM
- Key Characteristics of Sparse Attention Mechanisms
	1. Reducing compputational loads by zeroing out less important attention scores
	2. Leveraging PIM to accelerate sparse matrix operations


1. [CPSAA: Accelerating Sparse Attention using Crossbar-based Processing-In-Memory Architecture](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10374228)
	- This paper introduces CPSAA, a novel PIM-featured sparse attention accelerator that eliminates off-chip data transmissions. It presents an innovative attention calculation mode and a PIM-based sparsity pruning architecture, showcasing significant performance improvements and energy savings compared to traditional architectures.

2. [SparseP: Towards Efficient Sparse Matrix Vector Multiplication on Real Processing-In-Memory Systems](https://dl.acm.org/doi/pdf/10.1145/3508041)
	- SparseP provides a comprehensive analysis of Sparse Matrix Vector Multiplication (SpMV) on real-world PIM architectures. It offers insights into efficient SpMV algorithms tailored for PIM systems, covering various sparse matrices with diverse sparsity patterns.


