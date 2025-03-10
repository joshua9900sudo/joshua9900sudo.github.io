---
#title: "Sample Post"
date: 2025-03-03 21:55:00  # Created date
last_modified_at: 2025-03-04 21:38:00  # Modified date
---

# Summary of "Inherently Interpretable Physics-Informed Neural Network for Battery Modeling and Prognosis"  
This summary was conducted during an internship at IODA(Hanyang University).  
IODA Lab. : 

# Summary
## 0. Abstract  
- Q : Difference between the existing 'Data-Driven methods' and 'Model-based methods'.

## 1. Introduction
- The lithium-ion batterie has been being important and commonly used.
- For modeling and prognosis of batteries, there is BMS
- BMS monitors SOC(state-of-charge), EOD(end-of-discharge),  
SOH(state-of-health) and RUL(remaining useful life).
- SOC & EOD is for the state estimation[^1] of batteries 
in **a charge/discharge cycle**.
- SOH & RUL is for prognosis[^1] in the **entire life cycle**.

### Definitions of SOC, EOD, SOH and RUL

### EOD prediction : Model-based VS. Data-driven methods
- EOD prediction fall into two categories:
	- Model-based methods : Describe battery degradation 
	using mathematical models.
		- Advantage : Relatively high precision accuracy
		- Disadvantage : Difficult to solve due to PDEs in models.
	- Data-driven approaches : Map input data to output data 
	w/o any physics
		- Advantage : easily map input to output w/o physics
		- Disadvantage : Requirement of large labeled data and
		high-performance HW,
		Time-consuming data collection tasks and 
		DL's 'black-box' nature(Users want explanations).

### Why models containing PDEs are difficult to solve?
- High computational costs
- Sensitive to initial & boundary conditions
- Nonlinear

### PINN(Physics-informed NN) = Model-based + Data-driven methods
- Thelen et al. [45] and Lui et al. [46]'s research
	- It tracks only 3 physics parameters.

- Shi et al.[47], Tian et al.[48]
	- Only combine physical knowledge to preprocess the data.
	- No structural changes in the NN.

- Nascimento et al.[49]
	- Hybrid modeling method. physics-informed RNN(PIRNN)
	- But not fully incoporated physics to the structure of the NN.

[^1] : Estimation approximates *current* values or states while Prognosis predicts *future* trends.
