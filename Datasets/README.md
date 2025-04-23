# Dataset Overview

This repository contains multiple datasets related to power systems, including time-series measurements and node attributes. Below is a summary of each file and its contents.

---

## File Descriptions

### 1. `PMU_P.xlsx`
- **Description**: Time-series data from Phasor Measurement Units (PMUs) recorded at 1-minute intervals.
- **Time Range**: `2019-01-01 00:00:00` to `2019-05-19 23:59:00`.
- **Columns**:
  - `time`: Timestamp.
  - `NODE_0` to `NODE_15`: Measurements (likely voltage or power values in arbitrary units) for 16 nodes.
- **Structure**: Each row represents a timestamp with corresponding node measurements.

---

### 2. `PMU_node_attr.xlsx`
- **Description**: Adjacency matrix for PMU nodes, indicating connections between nodes (binary or weighted).
- **Nodes**: `0` to `15` (matching `PMU_P.xlsx`).
- **Structure**:
  - Rows and columns represent nodes.
  - `100` indicates a connection (e.g., node `0` and `1` are connected).
  - `0` indicates no connection.
- **Example**: Row `0`, Column `1` = `100` implies a connection between node `0` and `1`.

---

### 3. `ENTSO_regional_data_hourly.xlsx`
- **Description**: Hourly regional power data (likely load or generation values) from the European Network of Transmission System Operators (ENTSO-E).
- **Time Range**: `2015-01-01 00:00:00` to `2020-07-31 23:00:00`.
- **Columns**:
  - `Date`: Timestamp.
  - `node 1` to `node 15`: Numerical values (likely in MW or MWh) for 15 nodes.
- **Structure**: Each row represents hourly aggregated data for regional nodes.

---

### 4. `ENTSO_node_attr.xlsx`
- **Description**: Node attributes for the ENTSO-E dataset, possibly representing connection capacities or weights between nodes.
- **Structure**:
  - Rows and columns correspond to nodes `0` to `14`.
  - Values indicate capacities (e.g., `683.608` between node `0` and `2`).
- **Example**: Row `0`, Column `2` = `683.608` implies a capacity of 683.608 units between node `0` and `2`.

---

### 5. `NYC_data_11_nodes.xlsx`
- **Description**: 5-minute interval power data for 11 nodes in New York City (NYC), likely representing regional demand or generation.
- **Time Range**: `01/01/2016 00:00:00` to `12/31/2019 23:55:00`.
- **Columns**:
  - `Time Stamp`: Timestamp.
  - `CAPITL`, `CENTRL`, ..., `WEST`: Measurements for 11 NYC regions (values in MW or similar units).
- **Structure**: Each row contains time-stamped data for NYC regions.

---

### 6. `NYC_node_attr.csv`
- **Description**: Node adjacency matrix for the NYC dataset, indicating connection weights between nodes.
- **Nodes**: `0` to `10` (corresponding to regions in `NYC_data_11_nodes_2.xlsx`).
- **Structure**:
  - Rows and columns represent nodes.
  - Values denote connection weights (e.g., `270` between node `0` and `1`).
- **Example**: Row `0`, Column `1` = `270` implies a connection weight of 270 units between node `0` and `1`.

---


## Notes
- **Temporal Granularity**: Data resolutions vary (1-minute, 5-minute, hourly).
- **Node Relationships**:
  - Nodes in `PMU_P.xlsx` and `PMU_node_attr.xlsx` are directly linked (0–15).
  - Nodes in `ENTSO_regional_data_hourly.xlsx` and `ENTSO_node_attr.xlsx` correspond to a separate system (nodes 1–15).
  - NYC datasets (`NYC_data_11_nodes.xlsx` and `NYC_node_attr.csv`) represent an independent system with 11 nodes.
- **Units**: Numerical values are unitless unless specified by the data source.

---

## Potential Use Cases
- Time-series analysis of power systems.
- Network topology studies.
- Load/generation forecasting.
- Cross-validation of regional vs. high-resolution data.

For questions or clarifications, contact the dataset provider.  