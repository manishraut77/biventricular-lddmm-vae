# Biventricular LDDMM–VAE

This project learns heart shape variation from **61 CT-derived heart meshes** and generates synthetic biventricular surfaces and tetrahedral volumes.

**CT case 55 (CT55)** is the fixed reference heart. LDDMM registers it to each subject and produces a momentum field on **1,404 shared control points**. A Graph β-VAE learns these fields. Generation decodes new momenta, deforms CT55, and uses MMG3D to create a flat basal cut with mesh quality checks.

The current model learns shape variation without age, sex, or BMI inputs.

## Setup

Use Python 3.10+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy vtk torch
```

Generation also requires **MMG3D**, available as `mmg3d` or `mmg3d_O3` on your `PATH`, or supplied through `--mmg3d-executable`. Use ParaView to review registration labels and inspect meshes.

The source hearts should have roughly consistent orientations and a flat basal closure. Generation expects a tetrahedral CT55 volume and interprets coordinates and distance tolerances in millimetres.

## Run the pipeline

Run these commands from the repository root. They use the repository script names; remove download suffixes such as `(1)` if present. Use fresh output folders for a new run. Explicit paths below connect the stages despite older defaults in some scripts.

### 1. Collect, center, align, and extract surfaces

Place the original meshes at paths such as:

```text
0.00_CT_original/ct_case_0001/05_mesh_ref/mesh-complete.mesh.vtu
```

Include cases `ct_case_0001` through `ct_case_0061`, then run:

```bash
python 01_collectmeshes.py
python 02_center_ct_vtu.py --input-dir 01.1_Meshes
python 03_rigid_align_ct_to_55.py --input-dir 02.1_CenteredCT
python 04_extract_ct_surfaces.py --input-dir 03.1_ICPAlignedCT55
```

The collector writes `01.1_Meshes/ct_case_XXXX.vtu`. If these 61 flat input files are already available there, skip collection. Centering uses the boundary's area-weighted centroid; rigid ICP aligns each heart to CT55 using rotation and translation while retaining its size.

### 2. Register all hearts to CT55

First prepare the surface labels:

```bash
python 05_register_ct55_all.py all \
  --surface-dir 04.1_CTRegistrationSurfaces \
  --registered-dir 05.1_CTRegistered \
  --momenta-dir 05.2_Momenta
```

In ParaView, open the `*_review_labels.vtk` files under `05.1_CTRegistered/RegistrationMetadata/Prepared/`. Check `BasalFace` for the artificial basal cap and `AnatomicalObject` for the three non-basal surface components. Positive object IDs are assigned by surface area, rather than fixed anatomical names.

After reviewing the labels, run:

```bash
python 05_register_ct55_all.py all \
  --surface-dir 04.1_CTRegistrationSurfaces \
  --registered-dir 05.1_CTRegistered \
  --momenta-dir 05.2_Momenta \
  --reviewed-base-labels
```

Registration matches the non-basal objects and their basal contours under one shared deformation, with an endpoint flatness constraint. It publishes registered surfaces and momentum archives after registration QA passes.

### 3. Audit and train

```bash
python 06_fit_ct55_graph_vae_all.py audit \
  --momenta-dir 05.2_Momenta \
  --registration-dir 05.1_CTRegistered/RegistrationMetadata/Runs \
  --output 06.1_CT55GraphVAE_L16_H96/cohort_audit.json

python 06_fit_ct55_graph_vae_all.py train \
  --momenta-dir 05.2_Momenta \
  --registration-dir 05.1_CTRegistered/RegistrationMetadata/Runs \
  --output-dir 06.1_CT55GraphVAE_L16_H96 \
  --latent-dim 16 --hidden-dim 96 \
  --epochs 1800 --device auto
```

This configuration uses **16 latent dimensions and 96 hidden features**; the script defaults are 8 and 48. All 61 hearts are training data, so reported reconstruction metrics measure the training fit.

### 4. Generate a batch

```bash
python 07_generate_ct55_hearts_v4.py \
  --checkpoint 06.1_CT55GraphVAE_L16_H96/ct55_graph_momenta_vae_all61.pt \
  --momenta-dir 05.2_Momenta \
  --output-dir 07.1_Generated_LevelSet_v4_Batch100 \
  --count 100
```

This requests 100 samples. The default sampling policy is `aggregate-posterior`, with `--temperature 1.0`. The 61 training momentum files are needed to reconstruct that sampling distribution.

The v4 generator keeps each decoded momentum field unchanged during shooting. It then searches basal planes parallel to CT55's plane, uses MMG3D level-set insertion and local remeshing, and selects the shallowest tested candidate that passes QA. The final surface is extracted from the final volume. Local remeshing can change mesh connectivity.

## Math behind the model

The equations below follow the scripts with the run settings above and their remaining defaults.

### Momentum normalization

Let $N=61$, $M=1404$, and let $p_i\in\mathbb{R}^{M\times3}$ be subject $i$'s momentum field. The network uses:

$$
\bar p=\frac{1}{N}\sum_{i=1}^{N}p_i,
\qquad
s_p=\sqrt{\frac{1}{3NM}\sum_{i=1}^{N}\lVert p_i-\bar p\rVert_F^2},
\qquad
x_i=\frac{p_i-\bar p}{s_p}.
$$

Here, $\lVert \cdot\rVert_F^2$ sums the squared entries of a field. The mean is a full $M\times3$ field, and $s_p$ is one shared scalar. Control-point coordinates are separately centered and divided by one scalar RMS computed over all $3M$ centered coordinate components. These are invertible network transformations; heart size is retained.

### Graph encoder and decoder

Each node represents a shared control point, with six input features: three normalized coordinates and three normalized momentum values. The graph uses the **12 nearest non-self neighbors** per node. A graph convolution computes:

$$
g_v=W_{\mathrm{self}}h_v+b+
W_{\mathrm{neighbor}}\left(\frac{1}{12}\sum_{u\in\mathcal{N}(v)}h_u\right).
$$

Residual blocks apply layer normalization, GELU, and dropout to this update, then add a shortcut. Three encoder blocks and global mean/max pooling produce a latent mean $\mu_i$ and log variance $\ell_i$, with $\ell_i$ clipped to $[-10,6]$.

For an encoder input $u_i$:

$$
q_\phi(z\mid u_i)=\mathcal{N}\left(\mu_i,\mathrm{diag}(e^{\ell_i})\right),
\qquad
z_i=\mu_i+e^{\ell_i/2}\odot\epsilon_i,
\quad \epsilon_i\sim\mathcal{N}(0,I_d).
$$

Exponentials and $\odot$ act elementwise. Training uses noisy inputs $u_i=x_i+\eta_i$, where $\eta_i\sim\mathcal{N}(0,0.005^2I)$, while reconstruction targets stay clean. Evaluation uses $u_i=x_i$.

The decoder receives the latent vector at every node, together with normalized coordinates and sine/cosine position features at frequencies $1,2,4$, using arguments $\pi f c$. Two dense layers and two graph blocks predict a normalized field $\hat x_i=D_\theta(z_i)$. Physical momenta are recovered by $\hat p_i=\bar p+s_p\hat x_i$. Deterministic reconstruction uses $z_i=\mu_i$.

### Training loss

Let $B$ be the current batch size and $d$ the latent dimension. Reconstruction loss is the mean squared error over the batch, nodes, and three momentum components:

$$
\mathcal L_{\mathrm{rec}}=
\frac{1}{3BM}\sum_{i=1}^{B}\lVert \hat x_i-x_i\rVert_F^2.
$$

For KL regularization, the code first averages each latent dimension over the batch, then applies a floor of 0.1 nats before summing:

$$
K_j=\frac{1}{2B}\sum_{i=1}^{B}
\left(\mu_{ij}^2+e^{\ell_{ij}}-1-\ell_{ij}\right),
\qquad
\mathcal L_{\mathrm{KL}}=\sum_{j=1}^{d}\max(K_j,0.1).
$$

This is the script's free-bits rule: dimensions below the floor receive no additional gradient pressure from the KL term.

The aggregate-posterior moment penalty uses the batch mean and covariance, including each subject's posterior variance:

$$
m=\frac{1}{B}\sum_{i=1}^{B}\mu_i,
\qquad
C=\frac{1}{B}\sum_{i=1}^{B}(\mu_i-m)(\mu_i-m)^T
+\mathrm{diag}\left(\frac{1}{B}\sum_{i=1}^{B}e^{\ell_i}\right),
$$

$$
\mathcal L_{\mathrm{moment}}=
\frac{\lVert m\rVert_2^2}{d}+\frac{\lVert C-I_d\rVert_F^2}{d^2}.
$$

With the default loss settings, the objective at epoch $t$ is:

$$
\mathcal L=
\mathcal L_{\mathrm{rec}}+
\beta_t\mathcal L_{\mathrm{KL}}+
0.02\mathcal L_{\mathrm{moment}},
\qquad
\beta_t=0.0005\min\left(1,\frac{t}{600}\right).
$$

The moment penalty encourages mean zero and covariance $I_d$; it does not enforce a Gaussian distribution. Training uses AdamW, batch size 16, learning rate $5\times10^{-4}$, weight decay $10^{-4}$, cosine learning-rate decay to 5% of the initial rate, and gradient-norm clipping at 5.

### Generation and the flat basal cut

At evaluation time, encode all clean training fields to obtain the mixture:

$$
q_{\mathrm{agg}}(z)=\frac{1}{N}\sum_{i=1}^{N}q_\phi(z\mid x_i).
$$

Choose a subject index $J$ uniformly, sample its posterior, and apply temperature $T$ about the aggregate mean:

$$
z_0=\mu_J+e^{\ell_J/2}\odot\epsilon,
\qquad
m_{\mathrm{agg}}=\frac{1}{N}\sum_{i=1}^{N}\mu_i,
\qquad
z_T=m_{\mathrm{agg}}+T(z_0-m_{\mathrm{agg}}).
$$

Here $\epsilon\sim\mathcal{N}(0,I_d)$, and $T=1$ samples the fitted mixture. Decode $\hat p=\bar p+s_pD_\theta(z_T)$, then shoot the CT55 surface and volume with LDDMM.

For a candidate basal plane with unit normal $n$ and offset $a$, the generator evaluates the affine level set:

$$
\varphi(y)=r(n^Ty-a),\qquad r\in\{-1,+1\}.
$$

The sign selects the retained side. MMG3D inserts the interface $\varphi=0$ into the shot volume, with geometry away from the basal collar protected. Cap vertices are projected onto the plane and the result is checked. The saved momenta describe the LDDMM deformation before this mesh-cutting step.

## Outputs and quality checks

| Location | Contents |
| --- | --- |
| `05.1_CTRegistered/` | Registered surfaces, labels, and registration reports. |
| `05.2_Momenta/` | The 61 shared-control-point momentum archives. |
| `06.1_CT55GraphVAE_L16_H96/` | Checkpoint, cohort audit, metrics, and training history. |
| `07.1_Generated_LevelSet_v4_Batch100/GeneratedMomentas/` | Decoded momentum fields and sampling metadata. |
| `07.1_Generated_LevelSet_v4_Batch100/GeneratedSurfaces/` | Generated boundary surfaces (`.vtk`), when available. |
| `07.1_Generated_LevelSet_v4_Batch100/GeneratedVolumes/` | Generated tetrahedral volumes (`.vtu`), when available. |
| `07.1_Generated_LevelSet_v4_Batch100/Reports/` | Per-sample QA reports. |

Check `generation_summary.csv` or `generation_summary.json` for each sample's **ACCEPTED** or **REJECTED** status. Rejected attempts retain reports and momenta; a final mesh may be unavailable. A completed batch returns exit code `0` if every sample passes and `2` if any sample is rejected. Setup or other fatal errors return `1`.

Default acceptance thresholds include:

| Check | Requirement |
| --- | --- |
| Maximum basal-plane error | ≤ `1e-6` mm |
| Minimum tetrahedral scaled Jacobian | ≥ `0.02` |
| First-percentile scaled Jacobian | ≥ `0.10` |
| Minimum tetrahedral dihedral angle | ≥ `2°` |
| Basal contour loops | `3`: outer wall, LV opening, RV opening |

Additional checks cover positive, finite, nondegenerate tetrahedra; closed, edge-manifold surfaces; anatomical labels; protected geometry; and agreement with the reference cut surface.

All 61 subjects are used for training. Mesh QA measures the implemented geometric and numerical criteria; it does not establish generalization to unseen subjects or anatomical realism. Use each script's `--help` for further options.
