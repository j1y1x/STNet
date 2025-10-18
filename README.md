# STNet

### Basic Usage

train.py：run train.py to compute eigenvalues

plot_loss.py：draw the descending curve of the loss

predict.py：draw the image of the characteristic function

---

## How to Run Different Problems

### 1. Harmonic Problem

```bash
python train.py --d 2 --problem_type Harmonic --device cpu
```

or use GPU：

```bash
python train.py --d 2 --problem_type Harmonic --device cuda
```

### 2. Oscillator Problem

```bash
python train.py --d 2 --problem_type Oscillator --device cpu
```

or use GPU：

```bash
python train.py --d 2 --problem_type Oscillator --device cuda
```

### 3. Planck Problem

```bash
python train.py --d 2 --problem_type Planck --device cpu
```

or use GPU：

```bash
python train.py --d 2 --problem_type Planck --device cuda
```

- `--d` specifies the dimension of the problem.
- `--problem_type` selects the problem type (Harmonic, Oscillator, Planck).
- `--device` selects the computing device (cpu or cuda), automatically detected if not specified.

