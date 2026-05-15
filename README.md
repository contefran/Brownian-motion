# Diffusion Explorer

An interactive 3D diffusion measurement tool built with Streamlit. Simulates N independent Brownian particles, fits the diffusion coefficient D from the ensemble mean squared displacement (**MSD** — the average of squared distances from the origin across all particles at each time step), and lets you explore how the **⟨r²⟩ = 6Dt** scaling law emerges from randomness.

> Relevant to soft matter physics, drug delivery modelling, polymer science, and financial time-series analysis.

---

## Features

- **3D trajectory visualisation** — interactive Plotly scene, trail colour-coded by displacement, fully rotatable
- **Diffusion coefficient fitting** — linear regression on MSD, middle 20–80% segment to avoid edge artefacts
- **Per-walker analytics** — displacement over time, final displacement distribution, theoretical √N reference
- **Real-world context panel** — reframes the same simulation as nanoparticles, drug molecules, polymer chains, or financial assets
- **Simulation controls** — number of particles, steps, step size σ, random seed, trail resolution

## Stack

| Layer | Library |
|---|---|
| Simulation | NumPy (vectorised over all walkers) |
| Analysis | SciPy (`linregress`) |
| Visualisation | Plotly |
| App | Streamlit |

## Run locally

```bash
git clone https://github.com/contefran/Brownian-motion.git
cd Brownian-motion/brownian_app
pip install -r requirements.txt
streamlit run app.py
```

## Physics

**Sphere sampling.** Each step is a unit vector drawn uniformly on the 3D unit sphere. The naive approach — sampling the polar angle as φ ~ Uniform(0, π) — oversamples the poles, because the surface area element dA = sin φ dφ dθ is not uniform in φ. The correct fix is to sample cos φ ~ Uniform(−1, 1), which is area-preserving and produces a truly isotropic distribution.

**MSD scaling.** For a free random walk in 3D, the ensemble mean squared displacement grows linearly in time: ⟨r²⟩ = 6Dt, where D is the diffusion coefficient and t is the number of steps. This is Einstein's relation (1905), and the slope of the MSD curve is the observable that connects a microscopic random process to a macroscopic, measurable quantity — the same D that appears in Fick's second law.

**Fitting D.** The linear regression is run on the middle 20–80% of the MSD curve rather than the full range. Early steps have high variance because the ensemble hasn't had time to average out; late steps can show drift when the number of walkers is small. The middle segment is where the scaling is cleanest and the fit most reliable.

## Fields of application

**Soft matter and colloidal physics.** Brownian motion is the dominant transport mechanism for particles in the 1 nm–10 µm range suspended in a fluid. Measuring D from particle tracking experiments — via exactly this MSD approach — is a standard technique for characterising colloid size, solvent viscosity, and interaction forces.

**Drug delivery and biophysics.** The diffusion of nanoparticles and macromolecules through biological tissue is governed by the same ⟨r²⟩ = 6Dt law, modified by confinement and crowding. Deviations from linear MSD growth (anomalous diffusion, MSD ∝ tᵅ with α ≠ 1) signal obstacles, binding events, or active transport — all detectable from the MSD curve shape.

**Polymer science.** A freely-jointed polymer chain is mathematically identical to a 3D random walk: each monomer is a step, and the end-to-end distance scales as √N. The diffusion coefficient of the chain's centre of mass connects to its degree of polymerisation and the solvent quality via the Rouse and Zimm models.

**Quantitative finance.** The geometric Brownian motion model underpins the Black-Scholes equation and most option pricing frameworks. Log-price increments are modelled as i.i.d. Gaussian steps — a direct analogue of this simulation — and the MSD slope maps to volatility². Deviations from linearity are studied as signals of market inefficiency.
