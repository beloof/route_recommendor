# RouteRecommendor  

An intelligent route recommendation system that promotes active mobility (walking, cycling, etc.) by integrating geospatial, weather, and transport data.  

## Table of Contents  

- [Features](#features)  
- [Tech Stack](#tech-stack)  
- [Project Structure](#project-structure)  
- [Getting Started](#getting-started)  
- [Usage](#usage)  
  - [Graph Creation](#graph-creation)  
  - [Routes Dataset Generation](#routes-dataset-generation)  
  - [Prediction](#prediction)  
  - [PPO](#ppo)  
  - [DQN](#dqn)  
  - [Testing](#testing)  
- [Explanation and Theory](#explanation-and-theory)  
  - [Goal](#goal)  
  - [Prediction (GCN+GRU)](#prediction-gcn-gru)  
  - [PPO and DQN](#ppo-and-dqn)  
  - [Model Validation](#model-validation)  
- [Contributors](#contributors)  
- [License](#license)  

---

## Features  

- Multi-criteria route planning  
- Graph-based modeling of urban transport  
- AI-powered recommendation engine  
- Context-aware (weather, elevation, air quality)  

---

## Tech Stack  

- Python  
- NetworkX / PyTorch Geometric  
- OpenStreetMap, OpenWeatherMap, Data.gouv  
- GIS tools (Geopy, etc.)  

---

## Project Structure  

```

RouteRecommendor/
├── data/                         # Raw data
├── logs/                         # Training metrics and graphs
├── models/                       # Trained models
├── modules/                      # Utility scripts and notebooks
│   ├── classes.ipynb             # Data container classes
│   ├── constants.ipynb           # Global constants
│   ├── data\_cleaner.ipynb        # Data preprocessing
│   ├── environment.ipynb         # RL environment
│   ├── functions.ipynb           # Helper functions
│   ├── prediction\_model.ipynb    # Prediction model
│   ├── routes\_dataset\_generator.ipynb  # Synthetic route generator
│   ├── DQN.ipynb                 # DDQN implementation
│   ├── scoring\_model.ipynb       # PPO implementation
│   └── tester.ipynb              # Validation utilities
├── outputs/                      # Preprocessed data, embeddings, etc.
├── graph\_processor.ipynb         # Main pipeline (graph + models)
├── graph.ipynb                   # Graph creation
├── requirements.txt
└── README.md

````

---

## Getting Started  

1. Clone the repo  
2. Create a virtual environment  
3. Install dependencies:  
```
   pip install -r requirements.txt
````

---

## Usage

* Update **absolute paths** at the top of some notebooks if needed (relative paths already work).
* The main entry point is `graph_processor.ipynb`.
* For route generation: run `modules/routes_dataset_generator.ipynb`.
* For graph creation: run `graph.ipynb`.
* Parameters can be tweaked in `modules/constants.ipynb`.

### Graph Creation

Run `graph.ipynb`. It outputs:

* `outputs/graph_nx.json` – base graph of Strasbourg
* `outputs/graph_looped_nx.json` – graph with self-loops (used for prediction)
* `outputs/graph_mapping.json` – maps edge IDs between graphs

⚠️ Weather data is contextual (not static). Avoid running those cells unless you provide your API token in `modules/constants.ipynb`.

### Routes Dataset Generation

Run `modules/routes_dataset_generator.ipynb`. It outputs:

* `raw_routes.json` – unlabeled random routes
* `routes_dict.json` – labeled routes with CO₂, time, distance, inclination, etc.

You can configure:

* Number of random starting points
* Number of routes per origin

### Prediction

Run `graph_processor.ipynb`.
This trains the prediction model (`route_predictor.pth`) and evaluates prediction errors.

### PPO

Implemented in `graph_processor.ipynb` (PPO section). Produces a trained policy and reward plots.

### DQN

Implemented similarly to PPO. Trains a DDQN policy and logs rewards.

### Testing

* PPO & DQN → use `tester.ipynb` with a trained model and environment.
* Prediction model → tested in the last section of `graph_processor.ipynb`.

---

## Explanation and Theory

### Goal

Score and rank multiple routes from **best** to **worst** under multiple criteria.

---

### Prediction (GCN+GRU)

We predict key route attributes:

```
[co2e, distance, danger, time, traffic, light, ppl_density, freq, inclination]
```

Given weights:

$$
[W_{co2e}, W_{distance}, W_{danger}, W_{time}, W_{traffic}, W_{light}, W_{ppl}, W_{freq}, W_{inclination}]
$$

Score is computed as:

$$
\text{Score} = \sum (W_x \cdot x)
$$

**Pipeline:**

1. Build graph from GTFS + GBFS.
2. Enrich with APIs (weather, traffic, etc.).
3. Generate synthetic `raw_routes`.
4. Label routes → training/testing dataset.

---

### PPO and DQN

RL-based approach where a policy is learned over nodes, edges, and modes.

* Inputs: embeddings + labeled graph (CO₂, time, distance, etc.).
* Outputs: a ranked set of routes, balancing trade-offs (e.g., time vs CO₂).
* Rewards, embeddings, and actions handled in `environment.ipynb`.

---

### Model Validation

We validate models by reordering a pre-ranked (Pareto-based) set of routes and comparing with the reference order.


Metrics:

## Kendall’s Tau (τ)

Measures the correlation between two ranked lists by comparing concordant and discordant pairs:

$$
\tau = \frac{C - D}{\tfrac{1}{2}n(n-1)}
$$

Where:
- $C$ = number of concordant pairs (same order in both rankings)
- $D$ = number of discordant pairs (different order in both rankings)
- $n$ = number of items

## Spearman's ρ

Measures rank correlation using the squared differences between ranks:

$$
\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
$$

Where:
- $d_i$ = difference between ranks for each item
- $n$ = number of items

## Pairwise Accuracy (PA)

Measures the proportion of correctly ordered pairs:

$$
PA = \frac{1}{\tfrac{1}{2}n(n-1)} \sum_{j>i} \mathbf{1}[(r_i < r_j) \land (\hat r_i < \hat r_j)]
$$

Where:
- $r_i$ = true rank of item $i$
- $\hat r_i$ = predicted rank of item $i$
- $\mathbf{1}[\cdot]$ = indicator function (1 if condition true, 0 otherwise)

## Normalized Discounted Cumulative Gain (NDCG)

Measures ranking quality with emphasis on top results:

$$
DCG_k = \sum_{i=1}^k \frac{2^{rel_i}-1}{\log_2(i+1)}
$$

$$
NDCG_k = \frac{DCG_k}{IDCG_k}
$$

Where:
- $rel_i$ = relevance score of item at position $i$
- $k$ = number of items to consider
- $IDCG_k$ = ideal DCG (best possible ranking)

---

## Contributors

* **Zakariya Ghalmane** (Supervisor)
* **Badis Lassioued** (Intern)

---

## License

MIT License

