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

### EOD prediction : Model-based VS. Data-driven
- EOD prediction fall into two categories:
	- Model-based methods :
		- They describe battery degradation using mathematical models.  
		Because they consist of PDEs, difficult to solve.
	- Data-driven methods : 

### Why models containg PDEs are difficult?
- High computational costs
- Sensitive to initial & boundary conditions
- Nonlinear






[^1] : Estimation approximates *current* values or states while Prognosis predicts *future* trends.
