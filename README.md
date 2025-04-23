## FRGCN Model - Load Forcast
This project combines Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) for spatiotemporal prediction on urban data, aiming to learn the dynamic graph structure representing relationships between urban nodes.

# Key Features
•	Data Preprocessing: Reads urban node attributes (NYC_node_attr.csv) and time-series data (NYC_data_11_nodes.xlsx).
•	Dynamic Graph Construction: Generates a dynamic Laplacian matrix based on target cities and their associated cities.
•	Hybrid Model Training: Uses GCN and GAT layers for feature extraction, combined with Fractional Partial Differential Equations (FPDE) to simulate spatiotemporal evolution.
•	Result Saving: Outputs prediction results, loss values, and optimized parameters to Excel files.

# Environment Dependencies
•	Python 3.8+
•	Core Libraries:
bash
torch >= 1.10  
torch-geometric >= 2.0  
pandas >= 1.3  
numpy >= 1.21  
openpyxl  # For Excel file processing  

# Data Preparation
•	Node Attribute File: e.g., NYC_node_attr.csv
o	Format: Each row represents a city node, containing association strengths with other cities.
•	Time-Series Data: e.g., NYC_data_11_nodes.xlsx
o	Format: Each column corresponds to time-series observations of a city node.

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

# Code Structure
•	Model Definition: The GCNModel class includes GCN, GAT layers, and normalization operations.
•	Dynamic Weights: The gl_weights function generates edge weights based on node parameters.
•	Training Loop: Supports early stopping (patience=5) and learning rate decay.

# Example Output
Optimized alphas: L_j0: 0.8721, L_j1: 0.7563, L_j2: 0.6342  
Final MSE: 24.5732  

# License
Free to use with attribution required. Contact the developer for issues.
[file content end]

