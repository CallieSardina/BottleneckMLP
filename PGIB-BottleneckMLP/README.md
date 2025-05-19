## Setup

Follow original README instructions from originall PGIB repo for environmental setup & dataset download directions.

## Run

```
python -m models.train_gnns_bottleneckMLP --task mutag_128_32 --fc_dims 128 32 > out_mutag_128_32.out
```

train_gnns_bottleneckMLP.py uses the original train_gnns.py from PGIB with our modifications:

- IB Information Loss terms are reomoved. (Loss is only cross-entropy)
- We plug in our BottlenecMLP. Specify the dimensions/architecture of your desired MLP component using the flag --fc_dims (e.g. --fc_dims 128 32). 
- If you do not want to add the BottleneckMLP, use --fc_dims -1.
- MI values for each embedding layer (I(X; Z) and I(Z:Y)) will be saved to MI_logs directory. Please run mkdir ./MI_logs to first create this directory.

We provide additional files for our various experiemnts:
- train_gnns_nsa.py and train_gnns_convex_hull save the appropriate files for analysis of NSA/LNSA and convex hull volumes, respectively. 
- train_gnns_original.py is the original implementation of PGIB.

## Visualizations and Analysis

Use the following instructions do recreate the figures from our paper in your own experiments.

#### To recreate Figure 1:

```
./utils_edge_and_plots/plot_MI_embs.py
``` 
to plot information planes: I(X;Z) vs. I(Z;Y) over epochs.

#### To recreate Figure 2:

```
./for_KL_plot/plot_MI_XZ_graph.py
```
to plot I(X;Z) over epochs.

```
./for_KL_plot/plot_MI_ZY_graph.py
``` 
to plot I(Z;Y) over epochs.

#### To recreate Figure 3:

First run 
```
./similarity_utils/compute_nsa.py
``` 
to load embeddings (per category), and compute the NSA values.

Then run 
```
./similarity_utils/plot_nsa.py
```
to visualize the graphs.

#### To recreate Figures 5 & 6:

First run
```
./similarity_utils/convex_hull_pca.py
```
Then, 
```
./similarity_utils/plot_volumes_pca.py
```
