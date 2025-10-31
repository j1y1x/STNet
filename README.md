# STNet: Spectral Transformation Network for Operator Eigenvalue Problems (NeurIPS 2025)

[![STNet](docs/out.png)](docs/overview.png)

STNet is a learning-based approach for computing eigenvalues and eigenfunctions of linear operators. It trains a neural network under spectral transformations so that target eigenpairs become easier to learn, enabling stable multi-eigenpair training in one run.

![STNet overview](docs/overview.png)

---

## Highlights

- Deflation projection: remove already-learned eigensubspaces to avoid mode collapse and stabilize subsequent eigenpairs.
- Filter transform: reshape the spectrum (amplify a target interval, suppress outside) to improve convergence and accuracy.
- End-to-end multi-eigenpair learning: obtain several eigenpairs in one training process.
- Benchmarks covered: Harmonic, Schrödinger Oscillator, Fokker–Planck.

---

## Method

Let $L$ be a linear operator and $v_i$ the i-th eigenfunction. STNet represents $v_i$ with an MLP and applies two transforms each iteration:

1) Deflation: $D_i(L) = L − Q_{i−1} Σ_{i−1} Q_{i−1}^T$, where $Q_{i−1} = [ṽ_1, …, ṽ_{i−1}]$ and $Σ_{i−1} = diag(λ̃_1, …, λ̃_{i−1})$, to remove the span of learned eigenfunctions.
2) Filter: $F_i(L) = ∏_{j=1}^{i−1} [(L − (\tilde{\lambda}_j − ξ) I)(L − (\tilde{\lambda}_j + ξ) I)]$, which enlarges spectral gaps near target eigenvalues.

A residual-style loss encourages $v_i$ to match the action of the transformed operator without explicit inverses.

---

## Environment

- Python ≥ 3.9
- PyTorch
- CPU or CUDA GPU

---

## Quick Start

From repo root:

    # 2D Harmonic on GPU
    python train.py --d 2 --problem_type Harmonic --device cuda
    
    # 2D Harmonic on CPU
    python train.py --d 2 --problem_type Harmonic --device cpu

Other problems:

    # Schrödinger Oscillator
    python train.py --d 2 --problem_type Oscillator --device cuda
    
    # Fokker–Planck (named Planck)
    python train.py --d 2 --problem_type Planck --device cuda

Common args:

- --d : problem dimension (e.g., 1/2/5)
- --problem_type : Harmonic | Oscillator | Planck
- --device : cpu | cuda

---------

## Loss Curves & Visualization

    # Plot training loss curves
    python plot_loss.py
    
    # Visualize learned eigenfunctions
    python predict.py

---



## Acknowledgements

  We would like to express our gratitude to all collaborators, fellow students, and anonymous reviewers for their valuable assistance. Special thanks are extended to [Kuan Xu](http://staff.ustc.edu.cn/~kuanxu/) for significant support.
  And we would like to thank the following open-source projects and research works: [PMNN](https://github.com/SummerLoveRain/PMNN_IPMNN) for model architecture

## Citation

 If you use SNeT in your research, please use the following BibTeX entry.
```
@misc{wang2025stnetspectraltransformationnetwork,
      title={STNet: Spectral Transformation Network for Solving Operator Eigenvalue Problem}, 
      author={Hong Wang and Jiang Yixuan and Jie Wang and Xinyi Li and Jian Luo and Huanshuo Dong},
      year={2025},
      eprint={2510.23986},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.23986}, 
}
```
