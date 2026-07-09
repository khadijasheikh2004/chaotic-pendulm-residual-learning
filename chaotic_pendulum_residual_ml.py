"""
Residual learning demo for a chaotic driven pendulum.

This implements the idea from Bai-style orbit prediction correction in a toy
setting: use a simple physics model as a predictor, then train a machine
learning model to predict the error of that model.

The important pattern is:

    corrected_next_state = baseline_next_state + ML_predicted_baseline_error

The ML model is not trying to learn all pendulum dynamics from scratch. It only
learns the residual left behind by a deliberately imperfect physics model.

Run:
    python chaotic_pendulum_residual_ml.py

Outputs:
    pendulum_residual_results.png
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class PendulumParams:
    """Parameters for a periodically driven damped pendulum.

    The equation used below is:

        theta_dot = omega
        omega_dot = -(g/L) sin(theta) - damping * omega
                    - cubic_drag * omega^3
                    + drive_amp * cos(drive_freq * t)

    The cubic drag term is set to zero for the baseline model, making it an
    intentionally incomplete model of the truth dynamics.
    """

    damping: float
    drive_amp: float
    drive_freq: float
    gravity_over_length: float = 1.0
    cubic_drag: float = 0.0


# "Truth" is the high-fidelity simulator. In a real application this could be a
# trusted but expensive simulator, laboratory data, or high-accuracy telemetry.
TRUTH = PendulumParams(
    damping=0.23,
    drive_amp=1.20,
    drive_freq=2.0 / 3.0,
    gravity_over_length=1.0,
    cubic_drag=0.015,
)

# "Baseline" is the cheap model whose errors we want to learn. It has biased
# parameters and omits cubic drag, so it is systematically wrong but still useful.
BASELINE = PendulumParams(
    damping=0.20,
    drive_amp=1.05,
    drive_freq=2.0 / 3.0,
    gravity_over_length=0.94,
    cubic_drag=0.0,
)


def wrap_angle(theta: np.ndarray | float) -> np.ndarray | float:
    """Map angles to [-pi, pi) so trajectories do not drift by whole turns."""

    return (theta + np.pi) % TWO_PI - np.pi


def angle_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest signed angle difference a - b."""
    return wrap_angle(a - b)


def pendulum_rhs(t: float, y: np.ndarray, params: PendulumParams) -> np.ndarray:
    """Evaluate the right-hand side of the pendulum ODE."""

    theta, omega = y
    dtheta = omega
    domega = (
        -params.gravity_over_length * np.sin(theta)
        - params.damping * omega
        - params.cubic_drag * omega**3
        + params.drive_amp * np.cos(params.drive_freq * t)
    )
    return np.array([dtheta, domega])


def rk4_step(t: float, y: np.ndarray, dt: float, params: PendulumParams) -> np.ndarray:
    """Advance one step with fourth-order Runge-Kutta integration.

    RK4 is used for the truth model to make it more accurate than the baseline
    Euler step. That difference is one source of predictable baseline error.
    """

    k1 = pendulum_rhs(t, y, params)
    k2 = pendulum_rhs(t + 0.5 * dt, y + 0.5 * dt * k1, params)
    k3 = pendulum_rhs(t + 0.5 * dt, y + 0.5 * dt * k2, params)
    k4 = pendulum_rhs(t + dt, y + dt * k3, params)
    out = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    out[0] = wrap_angle(out[0])
    return out


def baseline_step(t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """Cheap predictor: one Euler step using the biased baseline parameters."""

    # This is deliberately cheaper and wrong: one large Euler step with slightly
    # biased physical parameters.
    out = y + dt * pendulum_rhs(t, y, BASELINE)
    out[0] = wrap_angle(out[0])
    return out


def truth_step(t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """High-fidelity target model used to generate supervised labels."""

    # The "truth" model uses RK4 and includes a small unmodelled cubic drag term.
    return rk4_step(t, y, dt, TRUTH)


def make_features(t: float, y: np.ndarray, baseline_next: np.ndarray) -> np.ndarray:
    """Build ML inputs for predicting the baseline model's one-step error.

    The feature vector includes:

    - current angle as sin/cos, avoiding discontinuity at +/- pi;
    - current angular velocity;
    - forcing phase as sin/cos;
    - the baseline model's own next-state prediction.

    Including the baseline prediction mirrors the paper's idea: the simple model
    is itself a strong predictor, and ML only needs to learn how it is wrong.
    """

    phase = TRUTH.drive_freq * t
    return np.array(
        [
            np.sin(y[0]),
            np.cos(y[0]),
            y[1],
            np.sin(phase),
            np.cos(phase),
            np.sin(baseline_next[0]),
            np.cos(baseline_next[0]),
            baseline_next[1],
        ]
    )


def build_one_step_dataset(
    rng: np.random.Generator,
    n_trajectories: int,
    steps_per_trajectory: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate supervised training pairs.

    For each sampled state, we compute two next states:

    - `base_next`: what the cheap baseline model predicts;
    - `true_next`: what the high-fidelity truth model predicts.

    The label is the residual `true_next - base_next`. During deployment the
    learner predicts that residual and adds it back to the baseline prediction.
    """

    x_rows = []
    y_rows = []
    for _ in range(n_trajectories):
        # Randomize the starting phase and state so the learner sees a broad
        # sample of the forced pendulum's behavior.
        t = rng.uniform(0.0, 30.0)
        state = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-2.0, 2.0)])
        for _ in range(steps_per_trajectory):
            base_next = baseline_step(t, state, dt)
            true_next = truth_step(t, state, dt)
            x_rows.append(make_features(t, state, base_next))
            y_rows.append(
                [
                    angle_delta(true_next[0], base_next[0]),
                    true_next[1] - base_next[1],
                ]
            )
            state = true_next
            t += dt
    return np.vstack(x_rows), np.vstack(y_rows)


def fit_residual_model(train_x: np.ndarray, train_y: np.ndarray):
    """Fit the supervised residual model.

    `RBFSampler + Ridge` is a lightweight kernel-style regressor:

    - `StandardScaler` normalizes feature magnitudes.
    - `RBFSampler` maps inputs into random Fourier features, approximating an
      RBF kernel without solving a full kernel regression problem.
    - `Ridge` learns a regularized linear map from those features to the two
      residual components: angle error and angular-velocity error.

    This plays the same conceptual role as an SVM/SVR residual model while
    keeping the demo quick and dependency-light.
    """

    return make_pipeline(
        StandardScaler(),
        RBFSampler(gamma=0.9, n_components=350, random_state=4),
        Ridge(alpha=1e-3),
    ).fit(train_x, train_y)


def corrected_step(t: float, state: np.ndarray, dt: float, model) -> np.ndarray:
    """Predict one step with baseline physics plus learned residual correction."""

    base_next = baseline_step(t, state, dt)
    correction = model.predict(make_features(t, state, base_next)[None, :])[0]
    corrected = np.array(
        [
            wrap_angle(base_next[0] + correction[0]),
            base_next[1] + correction[1],
        ]
    )
    return corrected


def rollout(
    initial_state: np.ndarray,
    t0: float,
    dt: float,
    n_steps: int,
    model,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Roll out truth, baseline-only, and corrected trajectories side by side.

    The baseline trajectory feeds its own previous baseline state forward. The
    corrected trajectory feeds its own corrected state forward. This tests the
    learned correction in the realistic closed-loop setting where small errors
    can accumulate.
    """

    times = t0 + dt * np.arange(n_steps + 1)
    truth = np.zeros((n_steps + 1, 2))
    baseline = np.zeros_like(truth)
    corrected = np.zeros_like(truth)
    truth[0] = initial_state
    baseline[0] = initial_state
    corrected[0] = initial_state

    for i in range(n_steps):
        t = times[i]
        truth[i + 1] = truth_step(t, truth[i], dt)
        baseline[i + 1] = baseline_step(t, baseline[i], dt)
        corrected[i + 1] = corrected_step(t, corrected[i], dt, model)
    return times, truth, baseline, corrected


def state_error(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Compute a combined angle/velocity error for each time step."""

    dtheta = angle_delta(predicted[:, 0], truth[:, 0])
    domega = predicted[:, 1] - truth[:, 1]
    return np.sqrt(dtheta**2 + domega**2)


def main() -> None:
    """Train the residual model, evaluate it, and save a diagnostic figure."""

    rng = np.random.default_rng(10)
    dt = 0.05

    # Training data are one-step examples sampled from many short truth
    # trajectories. The model learns local corrections, not a global closed-form
    # solution to the chaotic pendulum.
    train_x, train_y = build_one_step_dataset(
        rng,
        n_trajectories=30,
        steps_per_trajectory=180,
        dt=dt,
    )
    model = fit_residual_model(train_x, train_y)

    # Evaluate on a fresh initial condition. The 12-second horizon is deliberate:
    # chaotic systems eventually diverge even when the model is very good, so the
    # useful question is whether correction extends the short-term forecast.
    initial_state = np.array([1.15, 0.25])
    times, truth, baseline, corrected = rollout(
        initial_state=initial_state,
        t0=5.0,
        dt=dt,
        n_steps=240,
        model=model,
    )

    baseline_error = state_error(baseline, truth)
    corrected_error = state_error(corrected, truth)
    horizon = times - times[0]
    early = horizon <= 12.0

    # These summary numbers should show the corrected model reducing average
    # error over the forecast window.
    print("One-step training samples:", len(train_x))
    print("Mean baseline error over first 12s:  ", baseline_error[early].mean())
    print("Mean corrected error over first 12s: ", corrected_error[early].mean())
    print("Final baseline error at 12s:         ", baseline_error[-1])
    print("Final corrected error at 12s:        ", corrected_error[-1])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # 1) Direct angle trajectory comparison.
    ax = axes[0, 0]
    ax.plot(horizon, truth[:, 0], label="truth", color="#1b1b1f", linewidth=2)
    ax.plot(horizon, baseline[:, 0], label="baseline", color="#d95f02", alpha=0.9)
    ax.plot(horizon, corrected[:, 0], label="baseline + learned error", color="#1b9e77")
    ax.set_title("Angle rollout")
    ax.set_xlabel("seconds")
    ax.set_ylabel("theta, wrapped")
    ax.legend()

    # 2) Error over time on a log scale, making small improvements visible.
    ax = axes[0, 1]
    ax.semilogy(horizon, baseline_error + 1e-8, label="baseline", color="#d95f02")
    ax.semilogy(horizon, corrected_error + 1e-8, label="corrected", color="#1b9e77")
    ax.axvline(12.0, color="#666666", linestyle="--", linewidth=1)
    ax.set_title("State error")
    ax.set_xlabel("seconds")
    ax.set_ylabel("sqrt(angle_error^2 + omega_error^2)")
    ax.legend()

    # 3) Phase portrait: how the rollout moves through (theta, omega) space.
    ax = axes[1, 0]
    ax.plot(truth[:, 0], truth[:, 1], label="truth", color="#1b1b1f", linewidth=2)
    ax.plot(baseline[:, 0], baseline[:, 1], label="baseline", color="#d95f02", alpha=0.8)
    ax.plot(corrected[:, 0], corrected[:, 1], label="corrected", color="#1b9e77", alpha=0.9)
    ax.set_title("Phase portrait")
    ax.set_xlabel("theta")
    ax.set_ylabel("omega")
    ax.legend()

    # 4) Training diagnostic: predicted vs true one-step angle residuals. Points
    # near the diagonal mean the learner has captured the baseline's local error.
    ax = axes[1, 1]
    ax.scatter(train_y[::30, 0], model.predict(train_x[::30])[:, 0], s=10, alpha=0.35)
    lim = np.max(np.abs(train_y[::30, 0])) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color="#444444", linewidth=1)
    ax.set_title("Learned one-step angle residual")
    ax.set_xlabel("true residual")
    ax.set_ylabel("predicted residual")

    output_path = Path(__file__).with_name("pendulum_residual_results.png")
    fig.savefig(output_path, dpi=160)
    print("Saved plot:", output_path)


if __name__ == "__main__":
    main()
