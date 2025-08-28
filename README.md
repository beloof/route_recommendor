# RouteRecommendor

An intelligent route recommendation system that promotes active mobility (e.g., walking, cycling) by integrating geospatial, weather, and transport data.

## Table of Contents

-   [Features](#features)
    
-   [Tech Stack](#tech-stack)
    
-   [Project Structure](#project-structure)
    
-   [Getting Started](#getting-started)
    
-   [Usage](#usage)
    
    -   [Graph Creation](#graph-creation)
        
    -   [Routes Dataset Generation](#routes-dataset-generation)
        
    -   [Prediction](#prediction)
        
    -   [RL](#rl)
        
-   [Explanation and Theory](#explanation-and-theory)

	- [Goal](#goal)
	
	- [prcess](#process)
	
-   [Contributors](#contributors)
    
-   [License](#license)
    

## Features

-   Multi-criteria route planning
    
-   Graph-based modeling of urban transport
    
-   AI-based recommendation engine
    
-   Context-aware (weather, elevation, air quality)
    

## Tech Stack

-   Python
    
-   NetworkX / PyTorch Geometric
    
-   OpenStreetMap, OpenWeatherMap, Data.gouv
    
-   GIS tools (e.g., Geopy)
    

## Project Structure

```
RouteRecommendor/  
├── data/                         # Contains raw data  
├── logs/                         # Contains training metrics and graphs for all models
├── models/                       # Trained models  
├── modules/                      # Secondary or rarely run scripts  
│   ├── classes.ipynb             # Classes for containing and loading data  
│   ├── constants.ipynb           # Global constants  
│   ├── data_cleaner.ipynb        # Processes the raw data  
│   ├── environment.ipynb       # RL environment  
│   ├── functions.ipynb           # Modular helper functions  
│   ├── prediction_model.ipynb    # Main prediction model  
│   ├── routes_dataset_generator.ipynb  # Generates synthetic routes
│   ├── DQN.ipynb                 # DQN model
│   ├── tester.ipynb              # contains functions to test the models' validity  
│   └── scoring_model.ipynb       # PPO model
├── outputs/                      # Preprocessed data, graphs, embeddings, etc.  
├── graph_processor.ipynb         # Runs the AI models  
├── graph.ipynb                   # Builds the graph from collected data  
├── requirements.txt  
└── README.md  

```

## Getting Started

1.  Clone the repo
    
2.  Set up a virtual environment
    
3.  Install dependencies:
    
    ```
    pip install -r requirements.txt
    ```
    

## Usage

-   Make sure to change the **absolute paths** at the beginning of some notebooks and save them.
    
-   **Relative paths** are already used within the code, so you don’t need to modify those.
    
-   You’ll mostly use `graph_processor.ipynb`. If you want to generate new routes, you can run `modules/routes_dataset_generator.ipynb` or `graph.ipynb` to rebuild the graph. if you want to start from the very raw data you can run `modules/data_cleaner.ipynb`
    
-   You can also adjust parameters in `constants.ipynb`.
    

### Graph Creation

Run `graph.ipynb` — this will create three files:

-   `outputs/graph_nx.json`: basic Strasbourg graph
    
-   `outputs/graph_looped_nx.json`: same graph with self-loops (dummy edges added)
    
-   `outputs/graph_mapping.json`: maps edge IDs from the first graph to the looped version (the looped graph is used for prediction, so we need to track the original edges)
    

**Do not run the weather cells** — weather is treated as contextual, not static graph data. It also requires a Mistral key, which has limited free usage. But if you insist on doing so, please add your API token in `modules/contants.ipynb`

### Routes Dataset Generation

Run `modules/routes_dataset_generator.ipynb` — this will generate two files:

-   `raw_routes.json`: unlabeled routes (used as input for scoring)
    
-   `routes_dict.json`: labeled version of `raw_routes.json` (adds CO₂, time, distance, weighted inclination, etc.)
    

You can customize:

-   The number of random starting points (end of 2nd cell under "generate routes")
    
-   The number of routes per point (end of 3rd cell under "generate routes")
    

### Prediction

Run `graph_processor.ipynb` 
After training, you'll get `route_predictor.pth`.  
You can then run the **test** section to get prediction scores for each label.


### PPO

Run `graph_processor.ipynb` there is a ppo part that will return a trained model as well as a graph showing the training metrics

### DQN
same as PPO

### TESTING
for ppo and dqn just load a model and an environnement and run test_model from `tester.ipynb`
for the prediction model since it does not take the same inputs and outputs testing is a bit harder, there is a seperate section at the end of `graph_processor.ipynb` that does just that 


## Explanation and Theory

### Goal

The goal is to **score routes**. Given `n` routes, the model ranks them from best to worst.

### prediction (GCN + GRU)

To do this, we first predict the main parameters we’ll use to compute the score. Currently, we predict:

```
[co2e, distance, danger, time, traffic, light, ppl_density, freq, inclination]
```

Given a weight vector:

$$
[W_{co2e}, W_{distance}, W_{danger}, W_{time}, W_{traffic}, W_{light}, W_{ppl-density}, W_{freq}, W_{inclination}]
$$

The score is computed as:

$$
score= \sum (W_X \cdot X)
$$

### Process

1.  The model starts with GTFS and GBFS files to build the base graph.
    
2.  It enriches the graph using APIs and other available resources.
    
3.  It needs a `raw_routes` dataset. See `outputs/raw_routes.json` for the expected format.  
    Currently, these routes are generated randomly.
    
4.  The script labels those routes, creating training/testing data for the prediction model.
 

### PPO (and DQN)  
  A different implemented approach is **RL-based planning**, where a policy is learned and the score of a route corresponds to its probability under that policy.

 In this approach, the model takes node, edge and mode embeddings (extracted from the first model). it also takes a labeled graph containing data like co2e, time ,distance etc. (check graph_nx.json for more details). It will then proceed to explore the graph and learn tradeoffs between the different objectives (time vs co2e emission for example). both ppo and DQN models are implemented in this way
 
 the rewards, mebeddings and possible actions are all handled by the environment. check `modules/environment.ipynb` for detail


### Model Validation  

We validate our models by reordering a pre-ranked set of routes and comparing the predicted order to the reference order.  
The following metrics are used:  

- **Kendall’s Tau (τ):**  
  $$
  \tau = \frac{C - D}{\tfrac{1}{2}n(n-1)}
  $$ 
  Measures concordance of pairs (+1 = perfect, -1 = reversed).  

- **Spearman’s ρ:**  
  $$
  \rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
  $$  
  Rank correlation (monotonic relationship).  

- **Pairwise Accuracy (PA):**  
  $$
  \text{PA} = \frac{1}{\tfrac{1}{2}n(n-1)} \sum_{i < j} \mathbf{1}[ (r_i < r_j) \land (\hat{r}_i < \hat{r}_j) ]
  $$  
  Proportion of pairs ordered correctly.  

- **NDCG (Normalized Discounted Cumulative Gain):**  
  $$
  \text{DCG}_k = \sum_{i=1}^k \frac{2^{rel_i} - 1}{\log_2(i+1)}, 
  \quad
  \text{NDCG}_k = \frac{\text{DCG}_k}{\text{IDCG}_k}
  $$  
  Evaluates ranking quality, prioritizing top results.  


## Contributors

-   Ghalmane Zakariya (supervisor)
    
-   Lassioued Badis (intern)
    

## License

MIT License
