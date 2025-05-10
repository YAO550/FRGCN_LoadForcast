## FRGCN Model - Load Forcast
This project combines Fractional Order Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) for spatiotemporal prediction on urban data, aiming to learn the dynamic graph structure representing relationships between urban nodes.

# Key Features
•	Data Preprocessing: Reads PMU node attributes (PMU_node_attr.xlsx) and time-series power data (PMU_P_1.xlsx).
•	Dynamic Graph Construction: Generates a dynamic Laplacian matrix based on target nodes and their associated nodes in the power network.
•	Hybrid Model Training: Uses GCN and GAT layers for feature extraction, combined with Fractional Partial Differential Equations (FPDE) to simulate spatiotemporal evolution.
•	Result Saving: Outputs prediction results, loss values, and optimized parameters to Excel files.


# Environment Dependencies
•	Python 3.12.4
•	Core Libraries:
bash
SciPy 1.13.1
NumPy 1.26.4
PyTorch 2.5.1
Torch-Geometric 2.6.1 
openpyxl  # For Excel file processing  

# Data Preparation
•	Node Attribute File: PMU_node_attr.xlsx
o	Format: Each row represents a PMU node, containing association strengths (edge weights) with other nodes in a 16x16 adjacency matrix.
o	Example: Row 0 indicates the connection strength of Node 0 with Nodes 1 (100), 4 (100), etc.
•	Time-Series Data: PMU_P_1.xlsx
o	Format: Each column corresponds to time-series active power measurements (in MW) of a PMU node, with timestamps at 1-minute intervals.
o	Structure: 16 nodes (columns B-Q) and 200160 time steps (rows 2-200161), covering a 139-day observation window.


# Usage
Parameter Configuration (modify within the script):
python
M = 3           # Number of nodes (target city + associated cities)  
Nt = 93         # Time steps  
target_city = 5 # Target city index (0-10)  
target_col = 6  # Data column index (1-11)  

Run the Script:
bash
python FRGCN.py  

# Output Files:
•	results_part6_*.xlsx: MSE, RMSE, and optimized Alpha parameters for each training group.
•	results_data_part6_*.xlsx: Detailed comparisons between predicted and true values, along with RMSE.

# Comparison with Baseline Models
The FRGCN model is compared with two state-of-the-art spatiotemporal models:
•   STGCN (Spatio-Temporal Graph Convolutional Network):
Employs fixed graph structures with temporal convolutions and Chebyshev polynomial filters. While effective for stationary spatial relationships, it lacks FRGCN's dynamic graph learning capability and attention mechanisms for evolving urban patterns.
•   GraphWaveNet:
Uses adaptive adjacency matrices and diffusion convolutions to capture long-range dependencies. Compared to FRGCN's hybrid GCN-GAT architecture with FPDE-guided evolution, GraphWaveNet exhibits limitations in modeling fine-grained spatiotemporal interactions through fractional differential operators.

Both baselines demonstrate inferior performance in our experiments on dynamic urban networks, particularly for multi-step prediction tasks requiring adaptive graph reconfiguration.


# Notes
•	Environment Variable: The script sets KMP_DUPLICATE_LIB_OK="TRUE" to avoid library conflicts on some systems.
•	Data Integrity: Ensure input data contains no missing values; an exception is triggered for insufficient data in the final group.
•	Hardware Requirements: GPU acceleration is recommended. Reduce group_size if encountering insufficient video memory.

# Reference
•	. Zheng, X. et al. PSML: A Multi-scale Time-series Dataset for Machine Learning in Decarbonized Energy Grids (Code), Zenodo, 
https://doi.org/10.5281/zenodo.5663995 (2021).
•	The
European dataset originates from the European Network of
Transmission System Operators for Electricity (ENTSO-E)
and has been made open-source on the Kaggle platform
•	https://github.com/VeritasYin/STGCN_IJCAI-18.git
•	https://github.com/nnzhan/Graph-WaveNet.git

# License
Free to use with attribution required. Contact the developer for issues.
[file content end]

