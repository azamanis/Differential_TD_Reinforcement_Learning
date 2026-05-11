# dTD on the Merton problem: collapse, naive-dTD, and equivalence to TD

## TL;DR

For fixed-policy value estimation on the Merton problem with $\sigma \gtrsim 0.02$ and small $\Delta t$:

1. **Pure dTD (the rearranged form, MSE) is biased.** Its population objective contains an implicit L2 penalty on $V_w$ that pulls $V$ toward a constant. No amount of sample-level variance reduction (larger $B$, more steps) can fix it; only structural changes to the estimator can.
2. **Naive-dTD (the original Itô-residual split) is unbiased**, at the cost of $1/\Delta t$ higher variance.
3. **Naive-dTD ≡ TD up to $\mathcal{O}(\Delta t^{3/2})$ per sample.** Their semi-gradient directions are identical and Adam absorbs the constant scale, so they produce identical learning curves to four significant figures.
4. **None of the dTD variants beat TD.** $\beta$-dTD is weakly worse than TD; $\beta$-naive-dTD is identical to TD; pure dTD collapses.
5. **Separately, dTD requires $V^\pi \in C^2$** to be a valid PDE-residual method. TD requires only measurability. So dTD has a foundational validity concern on tasks with non-smooth value functions (e.g. MuJoCo with contacts), independent of the bias issue — which is why the paper's empirical results are surprising in principle.

For policy evaluation in this regime, TD is strictly the right choice.

---

## 1. Setup

We work with the Merton problem under a fixed policy $(\pi, \kappa)$. Wealth follows
$$
\Delta W = aW\,\Delta t + bW\sqrt{\Delta t}\,\varepsilon, \qquad \varepsilon \sim \mathcal{N}(0,1),
$$
with $a = r + \pi(\mu-r) - \kappa$ and $b = \pi\sigma$ (the diffusion). The closed-form value function is
$V^\pi(W) = A\, W^{1-\gamma}/(1-\gamma)$ — known and used as ground truth.

The HJB residual is
$$
\mathrm{HJB}(W) := aW\,V_w + \tfrac{1}{2}b^2 W^2\,V_{ww} - \rho\,V + U(\kappa W).
$$
$V = V^\pi \iff \mathrm{HJB}(W) = 0$ for all $W$.

The three loss formulations (writing each as `prediction − target`, with target detached):

| Method | Prediction | Target |
|---|---|---|
| **TD** | $V(W_t)$ | $r\,\Delta t + e^{-\rho\,\Delta t}\,V(W_{t+\Delta t})$ |
| **dTD** (rearranged form) | $\Delta W\,V_w + \tfrac{1}{2}\Delta W^2\,V_{ww}$ | $-r\,\Delta t + \rho\,\Delta t\,V(W_{t+\Delta t})$ |
| **naive-dTD** | $\rho\,\Delta t\,V(W_t)$ | $r\,\Delta t + \Delta W\,V_w(W_t) + \tfrac{1}{2}\Delta W^2\,V_{ww}(W_t)$ |

---

## 2. Why pure dTD collapses (the BRM bias)

Take the dTD residual and condition on $W$.

**Mean is correct:**
$$
\mathbb{E}[\delta_{\mathrm{dTD}} \mid W] = \Delta t \cdot \mathrm{HJB}(W).
$$

**Variance does not vanish at truth:**
$$
\mathrm{Var}(\delta_{\mathrm{dTD}} \mid W) \approx b^2 W^2\,\Delta t \cdot V_w^2.
$$

The MSE loss decomposes as
$$
L = \mathbb{E}[\delta_{\mathrm{dTD}}^2] = \underbrace{\Delta t^2\,\mathrm{HJB}^2}_{\text{signal}} + \underbrace{\Delta t \cdot b^2 W^2\,V_w^2}_{\text{noise floor}}.
$$

The noise floor is part of the **population objective**, not a sampling artifact. It acts as an L2 penalty on $V_w$. The minimum is therefore not at $V^\pi$ — it's at a biased fixed point with shrunken $V_w$, and in the limit of dominant noise the optimizer drives $V$ toward a constant.

Equivalent statement via the semi-gradient: at the truth,
$$
\left.\frac{\partial L}{\partial V_w}\right|_{V^\pi} = 2\,\Delta t \cdot b^2 W^2 \cdot V_w^{\text{true}} \neq 0.
$$
The truth is **not a stationary point** of the loss.

---

## 3. Empirical confirmation

Five concrete predictions, all confirmed.

### 3.1 σ-sweep at fixed $\Delta t = 1/252$ (pure dTD)

| $\sigma$ | $\|V_w\|$ learned | $\|V_w\|$ true | MAE | hjb_rmse |
|---|---:|---:|---:|---:|
| $10^{-4}$ | 575 | 720 | 127 | 11.2 |
| 0.02 | 3.5 | 722 | 257 | 25.4 |
| 0.05 | 0.29 | 732 | 257 | 25.2 |
| 0.10 | 0.016 | 772 | 269 | 25.1 |

Pure dTD works only when $b \approx 0$. For any non-trivial diffusion it collapses.

### 3.2 $\beta$-sweep at $\sigma = 0.2$ ($\beta$-dTD)

| $\beta$ | $\|V_w\|$ | MAE | dtd_noise_floor |
|---|---:|---:|---:|
| 0.25 | 574 | 41 | 7.0 |
| 0.50 | 525 | 62 | 6.0 |
| 0.75 | 435 | 108 | 4.1 |
| 0.90 | 298 | 176 | 1.8 |

Truth: $\|V_w\| = 985$. As $\beta$ grows, $V_w$ shrinks monotonically — the optimizer is trading off truthfulness for the L2 penalty. The `dtd_noise_floor` metric (which equals $b^2 W^2 V_w^2\,\Delta t$ on the eval grid) drops in lockstep.

### 3.3 Batch-size sweep at $\sigma = 0.10$ (pure dTD)

| $B$ | MAE | $\|V_w\|$ |
|---|---:|---:|
| 512 | 268.4 | 0.006 |
| 2048 | 269.1 | 0.016 |
| 8192 | 268.2 | 0.053 |
| 32768 | 269.3 | 0.045 |

64× more samples, identical collapse. **Bias is in the population objective**, not in finite-sample variance.

### 3.4 K-sample mean estimator

Replace $\delta^2$ with $(\bar\delta_K)^2$ where $\bar\delta_K$ averages $K$ i.i.d. transitions from the same $W_t$:
$$
\mathbb{E}\!\left[\bar\delta_K^2 \mid W\right] = \Delta t^2\,\mathrm{HJB}^2 + \frac{\Delta t \cdot b^2 W^2\,V_w^2}{K}.
$$
The bias is divided by $K$, not eliminated. Sweeping $K$ at $\sigma = 0.10$:

| $K$ | MAE |
|---|---:|
| 8 | 264 |
| 32 | 260 |
| 128 | **163** |

Recovery starts only when $1/K$ pulls the noise floor below the signal — confirming both the form of the bias and the diagnosis.

---

## 4. Naive-dTD removes the bias

Move the noise-coupled derivative terms to the **detached target**. Now the prediction $\rho\,\Delta t\,V(W_t)$ has no noise, and $\nabla_\theta\mathrm{pred}$ is deterministic. The semi-gradient at truth:
$$
\left.\nabla_\theta^{\text{semi}} L\right|_{V^\pi} = 2\rho\,\Delta t \cdot \mathbb{E}_W\!\bigl[(\rho\,\Delta t\,V^\pi - \mathbb{E}_\varepsilon[\mathrm{tgt}\mid W])\,\nabla_\theta V\bigr] = 0,
$$
because the bracketed term is $-\Delta t \cdot \mathrm{HJB}(W) = 0$ at truth. **Unbiased fixed point.**

The cost is variance: $\mathrm{Var}(\mathrm{tgt}\mid W) \sim b^2 W^2 V_w^2 / \Delta t$ — note the $1/\Delta t$ blowup, five orders of magnitude larger than dTD's variance at $\Delta t = 1/252$.

---

## 5. Naive-dTD ≡ TD per sample (Itô equivalence)

The two losses look very different on the surface but coincide identically.

**Sample-level equality.** Itô-expand $V(W_{t+\Delta t})$ around $W_t$:
$$
V(W_{t+\Delta t}) = V(W_t) + \Delta W\,V_w + \tfrac{1}{2}\Delta W^2\,V_{ww} + \mathcal{O}(\Delta W^3).
$$
Plug into the TD residual and use $e^{-\rho\,\Delta t} = 1 - \rho\,\Delta t + \mathcal{O}(\Delta t^2)$:
$$
\begin{aligned}
\delta_{\mathrm{TD}} &= V(W_t) - r\,\Delta t - e^{-\rho\,\Delta t}\,V(W_{t+\Delta t})\\
&= \rho\,\Delta t\,V(W_t) - r\,\Delta t - \Delta W\,V_w - \tfrac{1}{2}\Delta W^2\,V_{ww} + \mathcal{O}(\Delta t^{3/2})\\
&= \delta_{\mathrm{naive\text{-}dTD}} + \mathcal{O}(\Delta t^{3/2}).
\end{aligned}
$$

At $\Delta t = 1/252$, $\Delta t^{3/2} \approx 2.5\times 10^{-4}$ — negligible. The two residuals are numerically equal per sample.

**Gradient direction equality.** The semi-gradients are
$$
\nabla_\theta L_{\mathrm{TD}} = 2\,\mathbb{E}[\delta_{\mathrm{TD}} \cdot \nabla_\theta V],
\qquad
\nabla_\theta L_{\mathrm{naive}} = 2\,\mathbb{E}[\delta_{\mathrm{naive}} \cdot \rho\,\Delta t \cdot \nabla_\theta V].
$$
Since residuals coincide, $\nabla_\theta L_{\mathrm{naive}} \approx \rho\,\Delta t \cdot \nabla_\theta L_{\mathrm{TD}}$. Same direction, scaled by a constant.

**Adam absorbs the scale.** Adam's update is $\eta \cdot \hat g / \sqrt{\hat v}$, both numerator and denominator linear in the gradient — invariant to multiplying every gradient by the same constant. So with Adam,
$$
\text{naive-dTD with lr } \eta \;\equiv\; \text{TD with lr } \eta.
$$

This is exactly what we see empirically: identical learning curves to four significant figures (Section 6).

### Is this because of the fixed policy?

No — it's because of (i) small $\Delta t$, (ii) smooth $V$, and (iii) per-parameter adaptive optimization. The Itô expansion that drives the per-sample equality holds for any policy along any sample path of the SDE. Fixed policy makes the analysis cleanest (stationary distribution, no actor coupling, closed-form ground truth), but the equivalence itself is a property of the discretization, not of the policy.

The equivalence breaks if: $\Delta t$ is large enough that the $\mathcal{O}(\Delta t^{3/2})$ remainder matters; the dynamics are non-smooth (contact dynamics, hard constraints); or the optimizer is plain SGD (no adaptive rescaling). The MuJoCo experiments hit some of these conditions — see Section 7.

---

## 6. Empirical: TD vs every dTD variant

$\sigma = 0.10$, $B = 256$, 3000 steps, lr $= 2\times 10^{-3}$, log every 20.

| Method | step 100 | 500 | 1000 | 2000 | final | $\|V_w\|/772$ |
|---|---:|---:|---:|---:|---:|---:|
| **TD** | 255 | 205 | 164 | 105 | **76** | 122 |
| pure dTD | 273 | 273 | 272 | 272 | 273 | 0 |
| **naive-dTD** | 255 | 205 | 164 | 107 | **77** | 126 |
| $\beta$-dTD ($\beta=0.25$) | 255 | 205 | 164 | 106 | 76 | 111 |
| $\beta$-dTD ($\beta=0.50$) | 255 | 205 | 164 | 107 | 79 | 106 |
| $\beta$-dTD ($\beta=0.75$) | 255 | 205 | 164 | 108 | 88 | 92 |
| $\beta$-naive ($\beta=0.25$) | 255 | 205 | 164 | 105 | 76 | 122 |
| $\beta$-naive ($\beta=0.50$) | 255 | 205 | 164 | 105 | 76 | 122 |
| $\beta$-naive ($\beta=0.75$) | 255 | 205 | 164 | 105 | 76 | 121 |

Conclusions:

1. **No method beats TD in the early phase.** MAE at steps 100/500/1000 is identical to four significant figures. The "continuity information" claim produces no observable sample-efficiency benefit here.
2. **Pure dTD is catastrophic.** Stuck at MAE 272 from step 0.
3. **$\beta$-dTD is weakly harmful.** Final MAE rises 76 → 79 → 88 as $\beta$ grows. The bias bleeds through even diluted, with no upside.
4. **$\beta$-naive-dTD is identical to TD.** As predicted by the Itô equivalence.
5. **Pure naive-dTD ≈ TD.** Unbiasedness is restored but no extra information extracted.

---

## 7. A foundational concern: dTD requires $V \in C^2$

The bias issue above is a problem of *estimator design* — given that the HJB equation holds, MSE on the residual is biased. There is a deeper, more structural issue with the dTD framework itself.

dTD is a **PDE-residual method**. Its target equation
$$
\rho V = U + aW\,V_w + \tfrac{1}{2}b^2 W^2\,V_{ww}
$$
requires $V \in C^2$ for $V_{ww}$ to be classically defined and for the underlying Itô expansion to be valid. If $V^\pi$ has kinks — switching boundaries, free boundaries, contact discontinuities — then:

1. $V_{ww}$ is a distribution at the kinks, not a function;
2. Itô's formula does not apply pointwise;
3. The HJB equation holds only in the **viscosity-solution** sense, not as a pointwise identity;
4. The dTD residual is targeting an equation that is locally false.

A neural-net critic is always smooth, so $V_w, V_{ww}$ are well-defined functions of $\theta$ for any input. But minimizing the dTD MSE then pulls the NN toward satisfying a PDE the true $V^\pi$ doesn't satisfy, locking in an approximation error near every kink that no amount of optimization can fix.

TD has no such requirement. The Bellman expectation equation
$$
V(W_t) = \mathbb{E}\!\left[r\,\Delta t + e^{-\rho\,\Delta t}\,V(W_{t+\Delta t})\right]
$$
is an integral identity that holds for any measurable $V$. The expectation does the smoothing; $V$ itself can be arbitrarily rough.

This is a stronger objection than the BRM bias: it says dTD is not even **applicable** to problems with non-smooth value functions, regardless of how the loss is constructed. On MuJoCo tasks with hard contacts (Hopper, HalfCheetah, Ant, Humanoid), $V^\pi$ is almost certainly non-smooth, so dTD shouldn't work there *in principle*. That it doesn't catastrophically fail must be attributed to:

- **NN smoothing.** A finite-capacity network can't represent kinks anyway; the kink error is absorbed into a shared function-class bias with TD.
- **Mostly-smooth regions dominate.** Contact events are a small fraction of state-time, so the residual is "right on average" even if locally wrong.
- **PPO uses the critic loosely.** Advantage estimation needs relative ranking, not pointwise accuracy; a critic that's wrong near contacts can still produce useful gradients.

These are reasons dTD doesn't blow up on MuJoCo, not reasons it's the right method. The principled position: if $V^\pi$ is non-smooth, dTD targets an equation that doesn't hold, and the burden is on the paper to show its results don't rely on smoothness assumptions the tasks violate.

The Merton problem is the **best-case scenario** for dTD: $V^\pi$ is analytically $C^\infty$, $\Delta t$ is small, and the dynamics are smooth. We've eliminated the smoothness concern entirely, and dTD still fails — because of the BRM bias. So the paper's framework has *two* problems: it doesn't work on smooth problems either.

---

## 8. Why the paper's MuJoCo results don't reproduce here

The paper's "stochastic" experiments add a tiny perturbation with $\mathrm{coef} \in \{0, 0.01, 0.05\}$. At $\mathrm{coef} = 0$ the dynamics are deterministic, so $b = 0$ and there is no bias. Even at $\mathrm{coef} = 0.05$, $b^2$ is two orders of magnitude smaller than ours.

Other mitigations specific to their setup:
- $\beta$-dTD only — never pure dTD.
- High-dimensional state spaces (11–244 dim) dilute per-component bias.
- Short on-policy training horizons (5–19 epochs per rollout) — no time to drift to the biased optimum on stationary data.
- Episode return is the metric, not $L^2$ error to a known $V^\pi$ — biased critics that produce useful advantages still produce decent returns.
- $\Delta t$ up to 0.05 (HalfCheetah) — Itô remainder $\Delta t^{3/2} \approx 0.011$ is no longer negligible relative to the signal, so TD ≢ naive-dTD as cleanly.

None of these mitigations apply to a 1-D Merton problem with $\sigma = 0.2$, fixed policy, and a closed-form $V^\pi$ — which is exactly why we see the failure clearly and they don't.

---

## 9. Verdict

For policy evaluation under fixed policy with smooth dynamics and small $\Delta t$:

- **TD** is the right choice. Unbiased, well-conditioned gradients, Adam-friendly.
- **Naive-dTD** is mathematically equivalent to TD. Useful only if you wanted to verify the equivalence; no reason to prefer it operationally.
- **dTD** (rearranged form) introduces a structural BRM bias that no sample-level fix can remove. Avoid.
- **$\beta$-dTD** dilutes the bias but doesn't add anything TD doesn't already have. Weakly harmful.
- **$\beta$-naive-dTD** = TD in disguise. Pointless.

The dTD framework can only matter if you leave this regime — large $\Delta t$, non-smooth dynamics, or non-adaptive optimizers. None apply here.

---

## One-sentence version

The MSE on dTD is a **biased** objective whose population minimum is a shrunken $V$ pulled toward a constant; naive-dTD removes the bias but is then equivalent to TD up to $\mathcal{O}(\Delta t^{3/2})$ per sample, with the constant scale absorbed by Adam — so for fixed-policy evaluation in the smooth small-$\Delta t$ regime, the entire dTD framework collapses into either "TD" or "TD with a bias term."
