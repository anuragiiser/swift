# Swift: counterfactual targets for entangled machine unlearning

Code for the anonymous submission *Overshooting the Oracle: Why Zero Forget Accuracy Is the Wrong Target for Entangled Unlearning*.

**Swift** matches the loss distribution of the forget set $D_f$ to that of a held-out calibration set $D_{cal}$
(a stop-gradient counterfactual target), while a one-sided Wasserstein constraint keeps the retain set close to a
cheap retain reference:

\min_{\theta} W_2^2\left(P_\ell(D_f;\theta), \operatorname{sg}\left(P_\ell(D_{cal};\theta)\right)\right) + \lambda \left[W_2^2\left(P_\ell(D_r;\theta), \operatorname{sg}\left(P_\ell(D_r^{probe};\theta_r^{probe})\right)\right)\right]_+
$\lambda$ is set by dual ascent against a tolerance calibrated to the sampling floor of the retain term, and the
returned checkpoint is chosen by constrained selection on held-out slices (never on the retrained model).

| folder | dataset / model | forget request | settings |
|---|---|---|---|
| `cifar100/` | CIFAR-100, ResNet-18 | *aquarium fish* within the *fish* superclass | class, instance |
| `toxigen/` | ToxiGen, RoBERTa-base | toxic LGBTQ sentences (trained as benign) | class, instance |
| `sbic/` | Social Bias Frames, RoBERTa-base | all posts from one source (r/meanjokes) | class (source), instance |
| `vggface2/` | VGGFace2 (540 identities), ResNet-50 | largest-face images of 10 subjects | face size |

Each dataset folder contains the data and set construction, training of the Original model and of the retrained
reference (Retrain), Swift, and a README with the commands. Swift is always evaluated next to Original and
Retrain with identical metrics. `cifar100/` is the reference implementation of the method and the other folders
are ports of it; the text and face folders additionally carry the ablations.

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.11, PyTorch 2.2.2 (CUDA 12.1), torchvision 0.17.2, Transformers 4.56.2, Datasets 5.0.1,
scikit-learn 1.8 and NumPy 1.26 on NVIDIA H100 GPUs. With PyTorch < 2.4, keep `transformers<4.57` and `numpy<2`.

## Data

- **CIFAR-100** is downloaded by torchvision.
- **ToxiGen** (`toxigen/toxigen-data` on Hugging Face) is gated; request access and run `huggingface-cli login`.
- **SBIC** must be obtained from its authors' distribution page; pass the archive to
  `sbic/prepare_sbic.py --tgz /path/to/SBIC.v2.tgz`, or set `SBIC_URL` and it downloads.
- **VGGFace2** must be obtained from its authors; `vggface2/prepare_vggface2.py` reads the original release
  (with bounding boxes) or folders of face crops.

## Evaluation

Swift starts from the Original model and is compared with Retrain (trained without $D_f$) on accuracy over the
forget set, the adjacent and remote retain sets and held-out test sets, plus a loss-threshold membership
inference attack. The summary score is the L1 deviation from Retrain over the accuracy columns.

## Ablations

`ablate_swift_<dataset>.py` runs one group of variants, each differing from Swift in one aspect only;
`collect_ablations.py <results dirs> --out-dir tables` turns the results into tables. The ablations are run on
`toxigen/`, `sbic/` and `vggface2/`.

| group | variants (* = Swift) |
|---|---|
| `lambda` | *adaptive, fixed λ ∈ {0, 0.1, 1, 10, 100} |
| `selection` | *constrained (Eq. 6), penalty score, off (last step) |
| `forget_distance` | *W2², squared difference of mean losses |
| `retain_distance` | *W2², squared difference of mean losses |
| `retain_mode` | *one-sided, two-sided |
| `stopgrad` | *stop-gradient on D_cal, gradient through D_cal |
| `cal_fraction` | *100%, 50%, 25%, 10% of D_cal |
| `retain_ref` | FitCheap probe vs θ0's own losses |
| `cal_pool` | *pooled reference, one batch per step |
| `refinements` (text) | *Swift, literal objective |
| `lr` | three learning rates |
