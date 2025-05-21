## Setup

Follow original README instructions from originall GSAT repo for environmental setup & dataset download directions.

## Run

```
python run_gsat.py --dataset mutag --backbone GIN_with_fc_extractor --cuda -1 --exp_type fc-64-48-32
```

run_gsat.py uses the backbone specified. 

--backbone GIN --exp_type normal: Runs original GSAT
--backbone GIN --exp_type noinfo: Runs original GSAT without info loss
--backbone GIN_with_fc_extractor --exp_type fc-...: Log the architecture in ..., change the code in src/models/gin to the desired BottleneckMLP architecture.

## Visualizations and Analysis

For plotting I(X;Z) and info loss run run_gsat_kl_plot.py For plotting I(X;Z) vs I(Z;Y) over layers, run run_gsat_layers_in_GIN.py with the command above. 
