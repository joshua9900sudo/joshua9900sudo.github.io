---
title: "AttAcc_paper_summary"
date: 2025-01-30 10:00:00  # Created date
last_modified_at: 2025-02-26 08:00:00  # Modified date
---


# AttAcc PIM accelerator Paper Summarization
- EEE4610 Capstone Design
- lab : DTL
- Professor : Prof. Eui-Young Chung
- ASPLOS__2024__AttAcc! Unleashing the Power of PIM for Batched Transformer-based Generative Model Inference


# 1. Introduction
TbGM 특히, Gen 단계의 attention 및 FC layer 전체 실행 시간의 대부분을 차지.  
기존 시스템은 TbGM의 attention layer을 효과적으로 처리x.  
-> AttAcc라는 PIM 기반 아키텍처를 제안, GPU+PIM => heterogeneous system 설계  

# 2. Background
GPT 모델은 Transformer의 decoder based.  
TbGM inference = Sum 단계, Gen 단계
attention layer은 Key, Value mat 저장, batching이 불가능, 성능 병목  
기존 GPU 시스템에서는 attention layer이 낮은 연산 밀도를 가져 비효율적임.  

# 3. Benefits and Limits of Batching for TbGM
* 이점 
	* FC layer의 batching-> weight matrix reuse-> 연산 밀도를 증가  
	* 높은 batch 크기에서 FC layer의 throughput이 88배 증가  
* 한계
	* attention batching 효과 없  
	* 각 request의 KV matrix 달라 batching으로 메모리 재사용x  
	* 메모리 용량 한계 및 SLO 때문에 batch 크기 증가가 제한됨.  
	* e.g. GPT-3 175B KV matrix == 18GB에 달하며, if batch == 256, SLO(50ms) 초과  

# 4. TbGM Accelerator Architecture (Design Space Exploration)
* AttAcc : attention layer을 가속화 PIM, 기존 GPU 시스템의 한계 해결  
* xPU : FC layer, AttAcc : attention layer  
* HBM3 기반 PIM -> high internal BW  
* AttAcc where? -> buffer die, bank-group, bank -> place at each bank  

# 5. AttAcc Implementation
GEMV 유닛과 Softmax 유닛 설계.  
GEMV 유닛은 HBM 내부의 각 bank에 배치하여 높은 병렬성  
Softmax 연산은 상대적으로 낮은 BW requirement, buffer die에 배치.  
AttAcc는 명령어 기반 프로그래밍 모델을 사용하여 xPU와 효율적으로 협력.  

# 6. Maximally Utilizing AttAcc
* Attention-level piplelining : [Attention includes GEMV and Softmax]
GEMV and Softmax  

* Head-level pipelining : FC layer과 attention layer을 동시에 실행하여 
활용도를 증가시킴.

* Batch-level pipelining : 두 개 이상의 batch를 동시에 처리하여 
GPU와 PIM을 효과적으로 활용.  
=> harmful b.c. it's effective only if batch size is large  

* Feedforward Co-processing : FC layer 일부를 AttAcc로 offload하여 병렬성을 극대화.

# 7. Evaluation
* 실험 결과, DGX+AttAcc 시스템은 기존 DGX 시스템 대비 
최대 2.81배 높은 성능과 2.67배 낮은 에너지 소비량을 달성함.
* attention layer의 병목을 해소하여 전체 TbGM 실행 시간을 단축.
* SLO가 30ms일 때, 기존 GPU 시스템은 batch 크기가 1로 제한되지만, 
DGX+AttAcc는 batch 크기를 증가시켜 최대 56배 높은 처리량을 달성.

# 8. Discussion
* Multi-Query Attention(MQA) 및 Grouped-Query Attention(GQA)과의 호환성 검토.
* 낮은 비트 정밀도(INT8) 모델에서도 AttAcc의 성능 향상 효과 유지.
* 기존 GPU 시스템을 확장하는 방법(예: CPU와의 협업, GPU 개수 증가)과 비교하여 비용 효율성이 뛰어남.

# 9.Related Work

# 10. Conclusion
* attention layer의 메모리 병목을 해결하기 위해 PIM 기반 AttAcc를 제안.
* GPU-PIM 이기종 시스템을 활용하여 기존 GPU 시스템 대비 성능과 에너지 효율을 대폭 향상.
* AttAcc는 대규모 TbGM 추론을 효과적으로 지원할 수 있는 차세대 아키텍처로 자리 잡을 가능성이 큼.

# 11. Acknowledgments



# Next Research Brainstorming

### GQA 혹은 MQA를 적용한 Systolic Array 기반 GEMV 유닛 설계 (Refer to Sec. 8)

- Tradeoff : MHA vs GQA, MQA

	AttAcc는 기존 MHA 방식에서 독립적인 KV를 활용
	
	GQA(Grouped-Query Attention) 및 MQA(Multi-Query Attention)는 KV를 공유   
	-> 메모리 용량과 대역폭 요구사항을 줄일 수 있음 / 다만, AttAcc의 높은 BW가 의미 없어짐
	
	Systolic Array 기반의 GEMV 유닛을 설계 시  
	-> KV matrices를 reuse하여 aggregate BW를 높일 수 있음 / Higher area cost


- 연구 방향

	기존 GEMV unit을 Systolic Array로 변경 후 성능 테스트  
	MQA를 구현한 뒤, 한계점을 확인하고 GQA를 통해 group size를 조절하며 optimization 진행

- 코드 변경

	```src/devices.py``` (GEMV 유닛 변경)
	
	FC 레이어, MatMul 연산 등 세부 함수를 수정하여 Systolic Array로 변경  
	(기존 방식) 각 head는 독립적인 KV matrices -> KV matrices 공유 가능   
	KV matrices를 reuse하여 BW를 높임  
	연산 분할 방식을 최적화하여, GQA/MQA 환경에서도 높은 PIM 연산 성능을 유지.  

	1. Compute Time 수정
	
		xPU._compute_time()   
		PIM._compute_time()  
	
	2. 메모리 이동 및 BW 관련
	
		xPU._get_traffic()  
		xPU._mem_time()  
		PIM._mem_time()  
	
	3. 타일링 최적화 (L1, L2 Cache 고려)
	
		xPU._get_optimal_tile()   

	```src/config.py``` (모델 설정 추가)

		기존 MHA 구조에서 GQA/MQA가 가능한 구조로 변경  
		Systolic Array mode가 가능하게 변경

- 테스트 할 것들
	- GQA/MQA 환경에서 AttAcc의 성능 변화(Memory & BW requirement change, Performance change)
	- Systolic Array 기반 GEMV 유닛 설계 제안(Aggregate BW, Area cost, Performance change)

- Tradeoff 분석
	- 에너지 소모
	- performance
	- SLO
	- Area cost
	- Resource Requirement  

	등을 고려하여 최적의 GQA 또는 모델을 도출


