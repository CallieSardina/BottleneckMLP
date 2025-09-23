## Setup

Follow original README instructions from originall GSAT repo for environmental setup & dataset download directions.

## Run

```
python run_gsat.py --save_embs --dataset mutag --backbone GIN --bottleneck_dim normal --gaussianize --max_gauss_var 0.1 --max_gauss_schedule fixed
```

run_gsat.py uses the backbone specified. 

--backbone GIN --bottleneck_dim normal: Runs original GSAT
--backbone GIN --bottleneck_dim noinfo Runs original GSAT without info loss
--backbone GIN --bottleneck_dim 16: Runs GSAT w/o Info Loss with BottleneckMLP, specify the architecture in bottleneck_dim.

## Visualizations and Analysis

For plotting I(X;Z) and info loss run run_gsat_kl_plot.py. For plotting I(X;Z) vs I(Z;Y) over layers, run run_gsat_layers_in_GIN.py with the command above. For node drift, linkage, and entropy experiments; first run run_gsat.py with --save_embs and use those embeddings to run plot_node_drift.py, plot_node_linkage.py, and run_entropy.py


