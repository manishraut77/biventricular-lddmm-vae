# Biventricular LDDMM–VAE

This project learns heart shape variation from 61 CT-derived meshes and uses it to generate new biventricular heart meshes.

CT case 55 is the reference heart. LDDMM (Large Deformation Diffeomorphic Metric Mapping) smoothly deforms this reference to match each subject. A graph variational autoencoder (VAE) then learns the deformation patterns, stored as **momentum fields**.

```text
CT meshes → alignment → LDDMM registration → graph VAE → new heart meshes
```

## Setup

Use Python 3.10+ and run these commands from the repository folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy vtk torch
```

Put the 61 input meshes in `0.00_CT_original/`, named `ct_case_0001.vtu` through `ct_case_0061.vtu`. They should use the same coordinate units and have roughly matching orientations. The pipeline expects a flat basal cap and a tetrahedral CT55 template volume. Data and trained models are not included in the repository.

If your meshes are under `CT_original/ct_case_XXXX/05_mesh_ref/mesh-complete.mesh.vtu`, run `python 01_collect_vtu_cohort.py` first. It collects them into `MixedCohort/`; use that as the input folder below.

## Run the pipeline

Run these steps in order. Keep the paths shown here because some scripts still have older folder names as defaults. Use new output folders when starting a fresh run.

### 1. Prepare the meshes

Center each heart, align it to CT55, and extract its surface:

```bash
python 02_center_ct_vtu.py --input-dir 0.00_CT_original
python 03_rigid_align_ct_to_55.py --input-dir 02.1_CenteredCT
python 04_extract_ct_surfaces.py --input-dir 03.1_ICPAlignedCT55
```

This writes centered volumes, aligned volumes, and triangular surfaces into the numbered output folders. The original meshes are kept unchanged.

### 2. Register the cohort

First, prepare the surface labels:

```bash
python 05_register_ct55_all.py all \
  --surface-dir 04.1_CTRegistrationSurfaces \
  --registered-dir 05.1_CTRegistered \
  --momenta-dir 05.2_Momenta
```

Open the `*_review_labels.vtk` files in `05.1_CTRegistered/RegistrationMetadata/Prepared/` with ParaView. Check `BasalFace` for the flat base and `AnatomicalObject` for the three remaining surface components. After reviewing them, run:

```bash
python 05_register_ct55_all.py all \
  --surface-dir 04.1_CTRegistrationSurfaces \
  --registered-dir 05.1_CTRegistered \
  --momenta-dir 05.2_Momenta \
  --reviewed-base-labels
```

This produces registered surfaces and one momentum file per subject. Every momentum field uses the same 1,404 control points.

### 3. Train the VAE

Check the registration results, then train:

```bash
python 06_fit_ct55_graph_vae_all.py audit \
  --registration-dir 05.1_CTRegistered/RegistrationMetadata/Runs \
  --output 06.1_CT55GraphVAE_L16_H96/cohort_audit.json

python 06_fit_ct55_graph_vae_all.py train \
  --registration-dir 05.1_CTRegistered/RegistrationMetadata/Runs \
  --output-dir 06.1_CT55GraphVAE_L16_H96 \
  --latent-dim 16 --hidden-dim 96 \
  --epochs 1800 --device auto
```

This example uses 16 latent dimensions and 96 hidden features. The script defaults are 8 and 48. **All 61 subjects are used for training**, so the reported metrics describe the training fit, not performance on unseen hearts.

### 4. Generate new hearts

```bash
python 07_generate_ct55_hearts.py \
  --checkpoint 06.1_CT55GraphVAE_L16_H96/ct55_graph_momenta_vae_all61.pt \
  --output-dir 07.1_Generated_L16H96_Fitted10 \
  --count 10
```

By default, the script samples from the learned subject distributions, decodes new momenta, and refines them to keep the base flat. It then deforms the CT55 surface and volume while preserving their mesh connectivity.

Check `generation_summary.csv` and `Reports/` for each sample's `ACCEPTED` or `REJECTED` status. Both are saved. A completed run with rejected samples returns exit code `2`.

### 5. Check diversity (optional)

```bash
python 08_diagnose_ct55_diversity.py \
  --checkpoint 06.1_CT55GraphVAE_L16_H96/ct55_graph_momenta_vae_all61.pt \
  --refined-dir 07.1_Generated_L16H96_Fitted10/RefinedMomentas \
  --output-dir 08.1_DiversityDiagnostics_L16H96
```

This checks how much variation remains after reconstruction, sampling, and refinement. It compares momentum fields; inspect the generated meshes too.

## Math behind training

The equations below follow [`06_fit_ct55_graph_vae_all.py`](06_fit_ct55_graph_vae_all.py). The model learns momentum fields: the deformation parameters that move CT55 toward each training heart. It does not optimize mesh coordinates directly during VAE training.

### 1. Normalize the momentum fields

There are $N=61$ hearts and $M=1404$ control points. Each heart has a momentum field $p_i \in \mathbb{R}^{M\times3}$. The script subtracts the average field and divides by one shared scale:

$$
\bar p = \frac{1}{N}\sum_{i=1}^{N}p_i, \qquad
s = \sqrt{\frac{1}{3NM}\sum_{i=1}^{N}\|p_i-\bar p\|_F^2}, \qquad
x_i = \frac{p_i-\bar p}{s}
$$

Here, $\|A\|_F^2$ means the sum of the squares of every entry in $A$. This puts the inputs on a convenient numerical scale. Control-point coordinates are separately centered and divided by their RMS scale for use inside the network.

### 2. Learn from neighboring control points

The graph connects each control point to its $k=12$ nearest neighbors. At node $v$, a graph layer combines its own features with the average of its neighbors' features:

$$
g_v = W_{\mathrm{self}}h_v + b
+ W_{\mathrm{neighbor}}\left(\frac{1}{k}\sum_{u\in\mathcal{N}(v)}h_u\right)
$$

The weights $W$ and bias $b$ are learned. Each graph block applies layer normalization, GELU, and dropout to $g_v$, then adds a shortcut from $h_v$ so the original information can pass through.

The encoder starts with six features per node: three normalized coordinates and three normalized momentum values. After three graph blocks, it combines the mean and maximum features across all nodes. Two output heads predict a latent mean $\mu_i$ and log variance $\ell_i$; the log variance is clipped to $[-10,6]$ for stability.

### 3. Sample and reconstruct

Instead of assigning each heart one fixed latent vector, the encoder defines a Gaussian distribution:

$$
q_\phi(z\mid x_i)=\mathcal{N}\!\left(\mu_i,\operatorname{diag}(\sigma_i^2)\right),
\qquad \sigma_i=\exp(\ell_i/2)
$$

During training, the model samples from it using:

$$
z_i=\mu_i+\sigma_i\odot\epsilon_i, \qquad
\epsilon_i\sim\mathcal{N}(0,I), \qquad
\hat x_i=D_\theta(z_i)
$$

Here, $\odot$ means element-wise multiplication. This sampling formula lets gradients flow back through $\mu_i$ and $\sigma_i$.

The decoder receives the same $z_i$ at every control point, together with that point's normalized coordinates and sine/cosine position features at frequencies $1,2,4$ (arguments $\pi f c$). Two dense layers and two graph blocks produce three momentum values per node. Converting back to the original units gives $\hat p_i=\bar p+s\hat x_i$.

### 4. Minimize three losses

Let $B$ be the current batch size and $d$ the latent dimension. **Reconstruction loss** measures how close the decoded field is to the clean input:

$$
\mathcal{L}_{\mathrm{recon}}=\frac{1}{3BM}\sum_{i=1}^{B}\|\hat x_i-x_i\|_F^2
$$

**KL regularization** encourages each heart's latent distribution toward a standard normal. The script first averages over the batch for each latent dimension, then applies a floor of 0.1 before summing:

$$
K_j=\frac{1}{2B}\sum_{i=1}^{B}
\left(\mu_{ij}^2+\sigma_{ij}^2-1-\log\sigma_{ij}^2\right),
\qquad
\mathcal{L}_{\mathrm{KL}}=\sum_{j=1}^{d}\max(K_j,0.1)
$$

This floor is called *free bits*: below it, the KL term gives no further pressure to reduce that dimension's information.

**Moment matching** encourages the combined latent distribution to have mean zero and covariance $I$. Its batch mean and covariance are:

$$
m=\frac{1}{B}\sum_{i=1}^{B}\mu_i, \qquad
C=\frac{1}{B}\sum_{i=1}^{B}(\mu_i-m)(\mu_i-m)^T
+\operatorname{diag}\!\left(\frac{1}{B}\sum_{i=1}^{B}\sigma_i^2\right)
$$

The first part of $C$ measures variation between hearts; the diagonal term adds each heart's predicted uncertainty. The penalty is:

$$
\mathcal{L}_{\mathrm{moment}}=\frac{\|m\|_2^2}{d}
+\frac{\|C-I\|_F^2}{d^2}
$$

With the default weights, the complete objective is:

$$
\boxed{\mathcal{L}=\mathcal{L}_{\mathrm{recon}}
+\beta_t\mathcal{L}_{\mathrm{KL}}+0.02\mathcal{L}_{\mathrm{moment}}},
\qquad
\beta_t=0.0005\min\left(1,\frac{t}{600}\right)
$$

Here, $t$ is the epoch. The KL weight grows over the first 600 epochs, giving the model time to learn reconstruction before applying the full regularization.

Training uses AdamW with batches of 16, initial learning rate 0.0005, and weight decay 0.0001. The learning rate follows a cosine schedule down to 5% of its initial value, and gradient norms are clipped at 5. Small Gaussian noise (standard deviation 0.005) is added only to encoder inputs; reconstruction targets stay clean. Reported deterministic reconstructions use $z=\mu$ instead of a random draw.

## Main outputs

| Folder | What it contains |
| --- | --- |
| `05.1_CTRegistered/` | Registered heart surfaces and registration reports. |
| `05.2_Momenta/` | The 61 momentum fields used for training. |
| `06.1_CT55GraphVAE_L16_H96/` | Model checkpoint, `metrics.json`, and `training_history.csv`. |
| `07.1_Generated_L16H96_Fitted10/` | Generated surfaces, volumes, refined momenta, and quality reports. |
| `08.1_DiversityDiagnostics_L16H96/` | Diversity results in JSON and CSV. |

For more options, use a script's `--help` flag, such as `python 06_fit_ct55_graph_vae_all.py train --help`. Generation supports `--resume` with the same inputs and settings after an interrupted run; incomplete sample outputs may need attention before resuming.

The pipeline is built around this CT55 cohort. Quality checks help identify mesh problems, but do not establish anatomical plausibility or performance on new subjects.
