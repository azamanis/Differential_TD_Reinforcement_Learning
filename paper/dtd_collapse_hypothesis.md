# Why pure dTD collapses on Merton: a clean explanation

## The setup

We have a **fixed policy** $(\pi, \kappa)$ on the Merton problem. Wealth follows
$$
\Delta W = aW\,\Delta t + bW\sqrt{\Delta t}\,\varepsilon, \qquad \varepsilon \sim \mathcal{N}(0,1),
$$
with $a = r + \pi(\mu-r) - \kappa$ and $b = \pi\sigma$ (the diffusion).

The dTD per-sample residual is
$$
\delta = \underbrace{\Delta W\, V_w + \tfrac{1}{2}\Delta W^2 V_{ww}}_{\text{prediction}} - \underbrace{\bigl(-r\,\Delta t + \rho\,\Delta t\, V(W + \Delta W)\bigr)}_{\text{target}}.
$$

## Two facts about $\delta$

Conditioning on $W$ and averaging over the noise $\varepsilon$:

1. **Mean is correct:**
$$
\mathbb{E}[\delta \mid W] = \Delta t \cdot \mathrm{HJB}(W),
$$
where $\mathrm{HJB}(W) = aW V_w + \tfrac{1}{2}b^2 W^2 V_{ww} - \rho V + U(\kappa W)$ is the PDE we want $V$ to satisfy. At the truth $V = V^\pi$, $\mathrm{HJB}(W) = 0$ for all $W$. ✓

2. **Variance is large and depends on $V_w$:**
$$
\mathrm{Var}(\delta \mid W) \approx b^2 W^2\, \Delta t \cdot V_w^2.
$$
Crucially, this **does not vanish at the truth** — at $V = V^\pi$, $V_w$ is large and so is $\mathrm{Var}(\delta)$.

## The hypothesis

The training loss is $L = \mathbb{E}[\delta^2]$. Decompose:
$$
L = (\mathbb{E}\delta)^2 + \mathrm{Var}(\delta) = \underbrace{\Delta t^2\,\mathrm{HJB}^2}_{\text{signal we want}} + \underbrace{\Delta t \cdot b^2 W^2\, V_w^2}_{\text{noise floor}}.
$$

The second term is **part of the population objective itself**. It acts exactly like an L2 penalty on $V_w$: minimizing $L$ minimizes both terms simultaneously, so the optimizer is incentivized to **shrink $V_w$ toward zero**. The minimum of $L$ is therefore not at the true value function — it's at a biased fixed point where $V_w$ is shrunk and $V$ is pulled toward a constant.

This is why MSE on dTD doesn't converge: **the squared-residual loss is biased**, with the bias pointing toward a constant $V$.

## What we observed in the experiments

The hypothesis predicts five concrete things, all confirmed:

1. **Pure dTD should fail when $b > 0$.** Pure dTD at $\sigma \in \{0.02, 0.05, 0.10\}$: $V_w$ collapsed to $\sim 0$ in all cases. ✓
2. **Pure dTD should work when $b \approx 0$.** $\sigma = 10^{-4}$: $V_w$ recovered $\sim 80\%$ of truth. ✓
3. **Mixing TD in ($\beta$-dTD) should partially anchor $V_w$.** $\beta$-sweep at $\sigma = 0.2$: $V_w$ shrinks monotonically as $\beta$ grows (574, 525, 435, 298 for $\beta = 0.25, 0.5, 0.75, 0.9$; truth is 985). ✓
4. **Larger batch size should not help.** $B \in \{512, 2048, 8192, 32768\}$: $V_w$ stayed at $\sim 0$ in all cases. ✓ (Bias is in the population loss, not a sampling artifact.)
5. **A K-sample mean estimator should reduce the bias by $1/K$.** Replacing $\delta^2$ with $(\bar\delta_K)^2$ has population value $\mathrm{HJB}^2 + (\text{noise floor})/K$. Sweeping $K \in \{8, 32, 128\}$: MAE $264 \to 260 \to 163$, recovery starts kicking in only when $1/K$ pulls the noise floor below the signal. ✓

## Why MuJoCo hides this

The paper's "stochastic" experiments add a tiny perturbation with coefficient $\in \{0, 0.01, 0.05\}$. At $\mathrm{coef} = 0$ the dynamics are deterministic and there is no bias. Even at $\mathrm{coef} = 0.05$, $b^2$ is two orders of magnitude smaller than ours. They also use $\beta$-dTD (never pure dTD), high-dimensional states (which dilute per-component bias), short on-policy training horizons (no time to drift to the biased optimum), and evaluate by episode return rather than $L^2$ error to a known truth. None of those mitigations apply to a 1-D Merton problem with $\sigma = 0.2$ and a closed-form $V^\pi$ — which is exactly why we see the failure clearly and they don't.

## Naive-dTD avoids the bias

The paper introduces an alternative split (Table 1):

| | Prediction (gradient flows) | Target (detached) |
|---|---|---|
| **dTD** | $\Delta W\, V_w + \tfrac{1}{2}\Delta W^2 V_{ww}$ | $-r\,\Delta t + \rho\,\Delta t\, V(W_{t+\Delta t})$ |
| **naive-dTD** | $\rho\,\Delta t\, V(W_t)$ | $r\,\Delta t + \Delta W\, V_w(W_t) + \tfrac{1}{2}\Delta W^2 V_{ww}(W_t)$ |

Naive-dTD puts the noise-coupled $V_w \cdot \Delta W$ term in the **detached** target. The prediction $\rho\,\Delta t\, V(W_t)$ has no noise, so $\nabla_\theta\mathrm{pred}$ is deterministic. The semi-gradient at the truth is
$$
\nabla_\theta^{\text{semi}} L \big|_{V^\pi} = 2\rho\,\Delta t \cdot \mathbb{E}_W\!\left[(\rho\,\Delta t\,V^\pi - \mathbb{E}_\varepsilon[\mathrm{tgt}\mid W])\,\nabla_\theta V\right] = 0,
$$
because the bracketed quantity is $-\Delta t\cdot \mathrm{HJB}(W) = 0$ at the truth. **Unbiased fixed point.** The cost is variance: $\mathrm{Var}(\mathrm{tgt}\mid W) \sim b^2 W^2 V_w^2 / \Delta t$ — five orders of magnitude larger than dTD's variance at $\Delta t = 1/252$.

## Empirical comparison: early-phase learning

We re-ran TD, dTD, naive-dTD, $\beta$-dTD, and $\beta$-naive-dTD at $\sigma = 0.10$ with $B = 256$, 3000 steps, lr $= 2\times 10^{-3}$, log every 20 steps. MAE at intermediate checkpoints:

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

Five conclusions:

1. **No method beats TD in the early phase.** MAE at steps 100/500/1000 is identical to four significant figures across every converging method. The paper's qualitative claim that "dTD makes more informative updates by using continuity information" produces no observable sample-efficiency benefit here.
2. **Pure dTD is catastrophic.** Stuck at MAE 272 from step 0 — confirms the BRM bias is structural, not transient.
3. **$\beta$-dTD is weakly harmful.** Final MAE rises 76 → 79 → 88 as $\beta$ grows 0.25 → 0.5 → 0.75. The dTD bias bleeds through even diluted, with no upside.
4. **$\beta$-naive-dTD is harmless but pointless.** Identical to TD at every $\beta$ tested. Naive-dTD's gradient direction matches TD's closely enough that mixing them does nothing.
5. **Pure naive-dTD ≈ TD.** Same learning curve, same final accuracy. Unbiasedness is restored, but no extra information is extracted.

## Verdict for fixed-policy evaluation

In this regime, the dTD machinery — in any of its forms — provides **no benefit** over standard TD. It either collapses (pure dTD), is identical to TD (naive-dTD, $\beta$-naive-dTD), or is slightly worse than TD ($\beta$-dTD with non-trivial $\beta$). The paper's empirical wins on MuJoCo PPO must come from interactions with the actor (changing data distribution, advantage estimation, etc.) that don't apply here. For policy evaluation, TD is strictly the right choice.

## One-sentence version

The MSE on dTD is a **biased** objective: its population minimum is **not** $V^\pi$ but a shrunken value function with $V_w$ pulled toward zero, because $\mathrm{Var}(\delta) \propto V_w^2$ acts as an implicit L2 regularizer on $V_w$ inside the loss, and this regularizer survives every form of sample-level variance reduction (large batches, more steps) — only changes to the **estimator structure** (K-sample mean averaging out the noise term, or paired double-sampling) can remove it.
