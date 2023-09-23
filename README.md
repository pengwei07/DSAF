# DSAF: A Dual-Stage Adaptive Framework for Numerical Weather Prediction Downscaling

While widely recognized as one of the most substantial weather forecasting methodologies, Numerical Weather Prediction (NWP) usually suffers from relatively coarse resolution and inevitable bias due to tempo-spatial discretization, physical parametrization process, and computation limitation. With the roaring growth of deep learning-based techniques, we propose the **D**ual-**S**tage **A**daptive **F**ramework (**DSAF**), a novel framework to address regional NWP downscaling and bias correction tasks. DSAF uniquely incorporates adaptive elements in its design to ensure a flexible response to evolving weather conditions. Specifically, NWP downscaling and correction are well-decoupled in the framework and can be applied independently, which strategically guides the optimization trajectory of the model. Utilizing a multi-task learning mechanism and an uncertainty-weighted loss function, DSAF facilitates balanced training across various weather factors. Additionally, our specifically designed attention-centric learnable module effectively integrates geographic information, proficiently managing complex interrelationships. Experimental validation on the ECMWF operational forecast (HRES) and reanalysis (ERA5) archive demonstrates DSAF's superior performance over existing state-of-the-art models and shows substantial improvements when existing models are augmented using our proposed modules.

## Data
The datasets utilized in this study are derived from the European Centre for Medium-Range Weather Forecasts (ECMWF) operational forecast (HRES) and reanalysis (ERA5) archive. For regional NWP downscaling, we construct a real-world dataset called **Huadong**, covering the East China land and sea areas. In this dataset, HRES data is employed as the predictive data, while ERA5 reanalysis data serves as the ground truth.

- HRES: https://confluence.ecmwf.int/display/FUG/HRES+-+High-Resolution+Forecast
- ERA5: https://cds.climate.copernicus.eu/cdsapp#!/home

**Dataset Details.** The Huadong dataset encompasses a latitude range from $26.8^\circ$ N to $42.9^\circ$ N and a longitude range from $112.6^\circ$ E to $123.7^\circ$ E. It comprises a grid of $64 \times 44$ cells, with each cell having a grid size of 0.25 degrees in both latitude and longitude. Notably, the Huadong dataset incorporates Digital Elevation Model (DEM) data to represent terrain information. The HRES and ERA5 data cover the period from January 3, 2020, to April 1, 2022, and include eight weather factors: surface pressure (`sp`), 2m temperature (`2t`), 2m dewpoint temperature (`d2m`), skin temperature (`skt`), 10m u component of wind (`10u`), 10m v component of wind (`10v`), 100m u component of wind (`100u`), and 100m v component of wind (`100v`).

## Code Usage Instructions

- Data processing: `data_processing/EC_data_processing/ERA5.py` and `data_processing/EC_data_processing/HRES.py`

- 2x task: `DSAF/2x_task/`

- 4x task: `DSAF/4x_task/`

## Appendix
In this work, we approximate the $\nabla$ operator of different orders using the fourth-order different difference operator, corresponding to the $5\times5$ convolution kernel, thus represents $Loss_{reg}$.

$$
Loss_{reg} = \nabla^2 p + \nabla \cdot (\mathbf{u} \cdot \nabla \mathbf{u}). 
$$

The fourth-order different difference kernels with the shape of $5 \times 5$ are given by

$$
K_{s, 1}=\frac{1}{12}\left[\begin{array}{ccccc}
0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 \\
-1 & 8 & 0 & -8 & 1 \\
0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0
\end{array}\right],
$$

$$
K_{s, 2}=\frac{1}{12}\left[\begin{array}{ccccc}
0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 \\
-1 & 8 & 0 & -8 & 1 \\
0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0
\end{array}\right]^{T},
$$

$$
K_{s, 3}=\frac{1}{12}\left[\begin{array}{ccccc}
0 & 0 & -1 & 0 & 0 \\
0 & 0 & 16 & 0 & 0 \\
-1 & 16 & -60 & 16 & -1 \\
0 & 0 & 16 & 0 & 0 \\
0 & 0 & -1 & 0 & 0
\end{array}\right],
$$

$K_{s, 1}$ and $K_{s, 2}$ are different kernels for the first derivative $\partial x$ and $\partial y$, respectively. $K_{s, 3}$ is the kernel for the second derivative. 
