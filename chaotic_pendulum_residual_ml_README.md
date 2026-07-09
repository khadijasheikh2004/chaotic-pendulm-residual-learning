# Chaotic Pendulum Residual Learning Demo

This is a small implementation of the idea used in Bai-style model-error
prediction:

1. Run a cheap, imperfect baseline physics model.
2. Run a higher-fidelity truth model.
3. Train a supervised learner on `truth_next - baseline_next`.
4. During rollout, predict with the baseline and add the learned correction.

In `chaotic_pendulum_residual_ml.py`:

- `truth_step(...)` is a driven damped pendulum with RK4 integration and a small
  cubic drag term.
- `baseline_step(...)` is a biased, cheaper Euler model with no cubic drag.
- `make_features(...)` gives the learner the current state, forcing phase, and
  baseline next-state prediction.
- `fit_residual_model(...)` trains `StandardScaler -> RBFSampler -> Ridge`.
- `corrected_step(...)` applies the learned residual to the baseline prediction.

The demo focuses on a 12-second rollout. For chaotic systems, the point is not
indefinite trajectory matching; the useful result is extending the short-term
forecast horizon by correcting systematic model error.

Latest verified run:

- One-step training samples: 5400
- Mean baseline error over first 12s: 0.3843
- Mean corrected error over first 12s: 0.0992
- Final baseline error at 12s: 0.3205
- Final corrected error at 12s: 0.2082
