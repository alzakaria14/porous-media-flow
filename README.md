# Porous Media Flow PINN GPU

Created by:

**Al Zakaria, S.Pd.**

Supervised by:

**Amar Vijai Nasrulloh, S.Si., M.T., Ph.D.**

This project simulates two-dimensional water flow in porous media using three approaches within a single notebook:

- Finite Difference Method (FDM) as the numerical reference solution.
- A purely data-driven Neural Network (NN) as a surrogate model without physics constraints.
- A Physics-Informed Neural Network (PINN) built with PyTorch as a machine learning approach constrained by the governing physics.

The project focuses on water flow in swamp soil or a shallow aquifer under multi-point rainfall infiltration forcing over `6 hours`, on a 2D domain, with NN and PINN training on GPU when CUDA is available.

## Overview

The main notebook in this project is [`porous_media_flow_pinn_gpu.ipynb`](./porous_media_flow_pinn_gpu.ipynb). It builds the full simulation pipeline from scratch:

1. environment configuration, reproducibility, and GPU detection,
2. definition of the physical domain and hydrologic parameters,
3. generation of a multi-point spatiotemporal rainfall pattern,
4. PDE solution using explicit 2D FDM,
5. construction of Neural Network and PINN models with PyTorch,
6. generation of dummy observation data from FDM results at hours `1`, `2`, and `3`,
7. training of the purely data-driven Neural Network,
8. training of the PINN with combined physics and data losses,
9. evaluation of predictions for hours `4` to `6` against the FDM reference solution,
10. snapshot visualization and GIF animation generation for all three methods.

In practice, this notebook can be used as:

- learning material for porous media flow,
- a demonstration of integrating a classical numerical solver, a data-driven NN, and a PINN,
- an initial template for data-based groundwater / soil-water flow experiments,
- a basis for developing inverse PINNs for parameter estimation from field observations.

## Current Experimental Focus

The current notebook setup is designed to answer the following question:

- if only dummy observation data are available during the first `3 hours`,
- while the physical simulation continues until `6 hours`,
- which method better predicts the next `3 hours`: `NN` or `PINN`?

In this workflow:

- `FDM` is used as the numerical reference,
- `NN` is trained only on dummy observation data and the initial condition,
- `PINN` is trained on dummy observation data plus PDE, initial-condition, and boundary-condition constraints,
- the main evaluation is performed at hours `4`, `5`, and `6`.

## Modeled Problem

The model solves a simple 2D diffusion equation for hydraulic head `h(x, y, t)`:

```math
S_s \frac{\partial h}{\partial t} - K \left(\frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2}\right) - R(x,y,t) = 0
```

with:

- `h` = hydraulic head,
- `S_s` = specific storage,
- `K` = hydraulic conductivity,
- `R(x,y,t)` = rainfall infiltration source term.

In the notebook's FDM implementation, the equivalent form is:

```math
\frac{\partial h}{\partial t} = D \left(\frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2}\right) + \frac{R}{S_s}
```

with `D = K / S_s`.

## Current Simulation Parameters

Default parameters embedded in the notebook:

| Parameter | Value | Description |
| --- | --- | --- |
| Domain | `50 m x 50 m` | 2D simulation area |
| Simulation duration | `6 hours` | Short rainfall event |
| Spatial grid | `51 x 51` | `1 m` resolution in each direction |
| Initial head | `1.0 m` | Uniform initial condition |
| Boundary head | `1.0 m` | Dirichlet boundary condition on all sides |
| `K_day` | `0.8 m/day` | Base hydraulic conductivity |
| `K` | `0.03333 m/hour` | Converted from `K_day` |
| `S_s` | `0.15` | Specific storage |
| Target time resolution | `0.10 hour` | Further adjusted by the stability condition |
| Rainfall points | `18 random points` | Multi-source infiltration forcing |

The notebook also computes `dt` based on the explicit 2D stability condition:

```text
rx + ry <= 0.5
```

so the FDM simulation remains within a safe numerical range.

## Notebook Contents in Detail

### 1. Library imports and GPU configuration

The notebook loads the following main libraries:

- `numpy`
- `matplotlib`
- `torch`
- `psutil`
- `imageio`

At this stage it also performs:

- seed setup for reproducible results,
- activation of `torch.set_float32_matmul_precision("high")`,
- device selection between `cuda` and `cpu`,
- display of RAM and VRAM information to help monitor resources.

### 2. Physical domain and hydrologic parameters

This section defines:

- domain size `Lx`, `Ly`,
- grid counts `Nx`, `Ny`,
- simulation duration `T_end`,
- parameters `K`, `S_s`, and `D`,
- coordinate grids `X`, `Y`,
- time array `t_arr`.

Because the notebook uses an explicit FDM solver, `dt` is computed from the numerical stability limit rather than only from the target time resolution.

### 3. Multi-point rainfall forcing

One key component of this project is the spatially nonuniform rainfall infiltration source term. Rainfall is modeled as a combination of several random rain cells with:

- random rain centers within the domain,
- different spatial spread widths (`sigma`),
- different cell intensities,
- a time-varying piecewise-linear temporal profile.

The implementation is available in two versions:

- `rainfall_source_np(...)` for the FDM solver,
- `rainfall_source_torch(...)` for the PINN residual.

With this design, the rainfall forcing is:

- spatially heterogeneous,
- time-varying,
- consistent between the numerical solver and the PINN model.

### 4. Initial and boundary conditions

Initial condition:

- uniform hydraulic head `1.0 m` across the entire domain.

Boundary condition:

- constant Dirichlet boundary condition `1.0 m` on all four sides of the domain.

The current model is intentionally kept simple so the experiment remains focused on the influence of rainfall forcing and the FDM vs PINN comparison.

### 5. FDM solver

FDM is used as the reference solution. The notebook:

- builds the solution tensor `h[t, x, y]`,
- computes the stability indicators `rx` and `ry`,
- performs explicit updates at interior nodes,
- applies Dirichlet boundaries at each time step.

Advantages of this approach in the project:

- easy to verify,
- directly aligned with the same PDE,
- usable for generating pseudo-observations for the PINN.

### 6. Neural Network and PINN architecture

The `Neural Network` and `PINN` models are built from an `MLP` backbone based on `torch.nn.Module` with these characteristics:

- `3` input dimensions: `x`, `y`, `t`,
- default hidden size `128` for `NN` and `96` for `PINN`,
- multiple fully connected hidden layers with `Tanh` activation,
- `1` output dimension: predicted `h`.

Inputs are normalized with respect to `Lx`, `Ly`, and `T_end` for more stable training.

The difference is:

- `Neural Network` is used as a purely supervised model without PDE residuals,
- `PINN` uses the physics residual as additional regularization.

The PINN PDE residual is computed using PyTorch autograd to obtain:

- first derivatives with respect to `x`, `y`, `t`,
- second derivatives with respect to `x` and `y`,
- the physics residual used as a main loss component.

### 7. Training data sampling

PINN training uses three groups of points:

- interior-domain points for the PDE loss,
- initial-time points for the initial condition,
- points on all domain sides for the boundary condition.

In addition, the notebook generates dummy observation data from the FDM results:

- `12` spatial sensors,
- observations at hours `1`, `2`, and `3`,
- a small bias per hour,
- small Gaussian noise.

These dummy data are used by both learning models:

- `NN` uses them as the main training data,
- `PINN` uses the same data as part of the data loss,
- predictions are then extrapolated to hours `4` to `6`.

### 8. Neural Network and PINN training

Training is performed in two branches:

1. `Neural Network`
2. `PINN`

Both use a two-stage optimization process:

1. `Adam` for initial optimization,
2. `LBFGS` for final refinement.

For the `Neural Network`, the total loss combines:

- initial-condition loss,
- observation-data loss.

For the `PINN`, the total loss combines:

- PDE residual loss,
- initial-condition loss,
- boundary-condition loss,
- observation-data loss.

Important default configuration used in the notebook:

| Component | Value |
| --- | --- |
| `NN epochs_adam` | `5000` |
| `NN epochs_lbfgs` | `250` |
| `epochs_adam` | `8000` |
| `epochs_lbfgs` | `300` |
| `n_int` | `2500` |
| `n_ini` | `1200` |
| `n_bnd` | `1200` |
| `lr` | `8e-4` |
| `w_pde` | `1.0` |
| `w_ini` | `25.0` |
| `w_bnd` | `25.0` |
| `w_data` | `30.0` |

The notebook also stores loss histories for visualization after training for both models.

### 9. Result evaluation

After training, `NN` and `PINN` are evaluated on the full grid at hours `4`, `5`, and `6`, then compared with the FDM solution. The notebook displays:

- FDM reference head contours,
- Neural Network head contours,
- PINN head contours,
- absolute error map `|NN - FDM|`,
- absolute error map `|PINN - FDM|`,
- `RMSE`, `MAE`, and `max absolute error` metrics,
- a summary of which model performs better over the next `3-hour` prediction horizon.

This section is important for assessing how well the data-driven and physics-informed models predict conditions after the observation window ends.

### 10. Visualization and animation

The notebook provides visualization utilities for:

- 2D field plots,
- loss-history plots,
- rainfall-point overlays,
- dummy-sensor overlays.

In addition, the notebook creates:

- PNG frames for FDM results,
- PNG frames for Neural Network results,
- PNG frames for PINN results,
- `fdm_animation.gif`,
- `nn_animation.gif`,
- `pinn_animation.gif`.

When executed fully, the notebook also creates these output directories:

- `frames_fdm/`
- `frames_nn/`
- `frames_pinn/`

## Repository Structure

The current repository contents are compact:

```text
.
|-- porous_media_flow_pinn_gpu.ipynb
|-- README.md
`-- LICENSE
```

After running the notebook completely, the repository may also contain additional outputs such as:

- `frames_fdm/`
- `frames_nn/`
- `frames_pinn/`
- `fdm_animation.gif`
- `nn_animation.gif`
- `pinn_animation.gif`

## How to Run

### 1. Prepare the Python environment

Python `3.10+` is recommended.

Install dependencies:

```bash
pip install numpy matplotlib torch psutil imageio notebook jupyter
```

If you want GPU support, install the PyTorch build that matches your CUDA version from the official PyTorch documentation.

### 2. Start Jupyter Notebook

```bash
jupyter notebook
```

Then open:

```text
porous_media_flow_pinn_gpu.ipynb
```

### 3. Execute cells in order

Recommended order:

1. imports and device configuration,
2. domain parameters,
3. rainfall, initial condition, and boundary condition,
4. FDM solver,
5. Neural Network and PINN models,
6. data sampling,
7. Neural Network training,
8. PINN training,
9. prediction evaluation for hours 4-6,
10. animation generation.

Running cells in sequence is important because many notebook global variables depend on each other.

## Generated Outputs

If the notebook is run completely, you will obtain:

- device, RAM, and VRAM information,
- rainfall infiltration distribution snapshots,
- FDM hydraulic head snapshots,
- Neural Network training-loss curves,
- PINN training-loss curves,
- Neural Network prediction snapshots,
- PINN prediction snapshots,
- `NN` error maps relative to FDM,
- `PINN` error maps relative to FDM,
- GIF animations of FDM evolution,
- GIF animations of Neural Network evolution,
- GIF animations of PINN evolution.

## Project Strengths

- Combines a classical numerical method, a purely data-driven model, and a PINN in one workflow.
- Rainfall forcing is reasonably realistic because it is multi-point and time-varying.
- Already prepared to leverage GPU for training.
- Provides pseudo-observations for data-driven simulation.
- Enables direct comparison between `NN` and `PINN` for prediction after the observation window ends.
- Provides fairly complete visualization for qualitative and quantitative analysis.

## Current Limitations

- The repository is still a single monolithic notebook and has not yet been split into Python modules.
- The observation data used for training are still dummy data and only cover hours `1-3`, not real field measurements.
- Soil parameters are still homogeneous and isotropic.
- Boundary conditions are still uniform on all sides.
- The physical model is still a simple diffusion model and does not yet include more complex hydrologic processes.
- Prediction evaluation for hours `4-6` still uses FDM as the internal ground truth rather than benchmarking against real observations.

## Future Directions

Further development already implied by the notebook includes:

- heterogeneous permeability `K = K(x, y)`,
- anisotropy `Kx != Ky`,
- river or canal boundaries,
- evapotranspiration,
- real rainfall data,
- replacement of dummy observations with piezometer / monitoring-well data,
- inverse PINNs for parameter estimation from observational data.

## Suggested Repository Refactor

If this project is developed further, the following structure will be easier to maintain:

```text
.
|-- notebooks/
|-- src/
|   |-- fdm_solver.py
|   |-- nn_model.py
|   |-- pinn_model.py
|   |-- rainfall.py
|   `-- visualization.py
|-- outputs/
|-- README.md
`-- LICENSE
```

With this separation, experiments will be easier to test, document, and reproduce.

## License

This project uses the [MIT](./LICENSE) license.
