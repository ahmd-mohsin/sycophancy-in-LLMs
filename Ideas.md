% ─────────────────────────────────────────────────────────────────────────────
% §3.0  Motivation and Intuition  (insert BEFORE §3.1 Dataset Construction)
% ─────────────────────────────────────────────────────────────────────────────

\subsection{Motivation: Why Scalar Rewards Fail and How Decomposition Fixes It}
\label{sec:method_motivation}

\paragraph{Sycophancy as an attention allocation problem.}
Transformer attention assigns weight $\alpha_{ij}$ to token $j$ when
generating token $i$. Let $\mathcal{A} \subset [T]$ denote the set of
\emph{authority tokens} (e.g., ``Nobel laureate'', ``world expert'') and
$\mathcal{F} \subset [T]$ the set of \emph{factual tokens} (e.g.,
``study shows'', ``data indicate''). Define the \emph{authority attention
ratio} at layer $\ell$ and head $h$ as:
\begin{equation}
  \label{eq:aar}
  \rho^{(\ell,h)} \;=\;
  \frac{\sum_{j \in \mathcal{A}} \bar{\alpha}^{(\ell,h)}_j}
       {\sum_{j \in \mathcal{F}} \bar{\alpha}^{(\ell,h)}_j + \varepsilon},
  \qquad
  \bar{\alpha}^{(\ell,h)}_j = \frac{1}{T}\sum_{i=1}^T \alpha^{(\ell,h)}_{ij}.
\end{equation}
Empirically, $\rho^{(\ell,h)}$ increases monotonically with pressure
level $j$ in pre-trained models (Figure~\ref{fig:attention_mass}): authority
tokens absorb disproportionately more attention than factual tokens as
pressure escalates, even when the factual content of the prompt is held
fixed. This is the mechanistic substrate of sycophancy — authority cues
override evidence by capturing the attention budget that would otherwise
flow to factual tokens.

\paragraph{Why RLHF cannot correct this via a scalar reward.}
Under RLHF, the policy is optimised against a scalar proxy reward
$\hat{R}(x,y)$ learned from human preferences.
Suppose $\hat{R}$ encodes an agreement bias (as shown in
Proposition~\ref{prop:misspec}).
The optimal policy then satisfies:
\begin{equation}
  \pi^*_{\hat{R}}(y \mid x) \;\propto\;
  \pi_{\mathrm{ref}}(y \mid x) \cdot
  \exp\!\left(\tfrac{1}{\beta} \hat{R}(x, y)\right).
\end{equation}
Because $\hat{R}$ does not distinguish between sycophantic agreement
(\emph{capitulating} to authority against evidence) and genuine agreement
(\emph{correctly} following strong evidence), both are reinforced equally.
A sycophantic completion $y_s$ that entails the pressured opinion and ignores
$C'$ receives $\hat{R}(x, y_s) \approx \hat{R}(x, y^*)$ for a correct
completion $y^*$, so the gradient provides no signal to separate them.
The KL penalty $\beta D_{\mathrm{KL}}$ cannot correct this: it constrains
the \emph{magnitude} of policy movement but not its \emph{direction}, which
is determined entirely by $\hat{R}$.

\paragraph{Reward hacking via length collapse.}
GRPO's within-group normalisation (Eq.~\eqref{eq:grpo_adv}) creates a second
failure mode: if all completions in a group have similar reward, the
advantages $\hat{A}_i \approx 0$ and the gradient vanishes.
Without a length floor, the policy discovers that very short completions
(e.g., a single hedge sentence) produce uniformly low but equal rewards
across the group, collapsing $\sigma_R \to 0$ and halting learning.
In GRPO v1 this manifested as mean completion length dropping below 60
tokens by epoch 0.8, with KL divergence spiking to 0.74 and PACF
regressing to $-0.42$ (Table~\ref{tab:training_dynamics},
Figure~\ref{fig:reward_hacking}).
The length multiplier $\lambda(y)$ (Eq.~\eqref{eq:length}) prevents this
by zeroing the reward for any completion under 60 words, ensuring
$\sigma_R > 0$ within each group.

\paragraph{Why decomposed rewards create the correct gradient.}
Let $y_s$ be a sycophantic completion (capitulates to $P_j$, ignores $C'$)
and $y^*$ a correct completion (resists $P_j$, follows $C'$).
Under our decomposed reward (Eq.~\eqref{eq:reward_total}):
\begin{align}
  R(y_s, C') &\approx
    (\alpha{+}\gamma) \underbrace{R_p(y_s, C')}_{\text{low: drifts from }b(C')}
    + \beta \underbrace{R_c(y_s, C')}_{\text{low: entails }o\text{, not }b(C')}
    + \varepsilon \underbrace{R_{\mathrm{pos}}(y_s, C')}_{\text{negative: contradicts }b(C')}
    - \delta \underbrace{R_g(y_s)}_{\text{high: agrees with }o}
    \;\ll\; R(y^*, C'), \label{eq:why_decomp}
\end{align}
so the group-normalised advantage $\hat{A}(y_s) \ll \hat{A}(y^*)$, producing
a strong negative gradient on $y_s$ and a positive gradient on $y^*$.
A scalar reward that gives both $y_s$ and $y^*$ similar scores produces
$\hat{A}(y_s) \approx \hat{A}(y^*) \approx 0$ — no gradient, no learning.
The decomposition therefore solves the reward distinguishability problem
directly: sycophantic and non-sycophantic completions are guaranteed to
receive different total rewards as long as at least one component disagrees,
which is ensured by the non-redundancy property established in
Section~\ref{sec:prelim_decomp_reward}.

\paragraph{The NLI gate as a variance guarantee.}
A subtler version of the same problem arises at the dataset level.
If the two baselines $b(C)$ and $b(C')$ are semantically similar
($\text{shift}(b(C), b(C')) < \tau$), then for any completion $y$:
\begin{equation}
  R_c(y, C) \;=\; p_{\mathrm{entail}}(b(C), y)
  \;\approx\; p_{\mathrm{entail}}(b(C'), y) \;=\; R_c(y, C'),
\end{equation}
and similarly for $R_p$ and $R_{\mathrm{pos}}$.
All six samples in the group receive approximately the same reward regardless
of context type, collapsing $\sigma_R \approx 0$ and zeroing the GRPO
advantage signal.
The NLI gate (Eq.~\eqref{eq:nli_gate}) is therefore not a data quality
heuristic but a \emph{mathematical precondition} for GRPO to produce a
non-zero gradient: it guarantees $\text{shift}(b(C), b(C')) \geq \tau$,
which in turn guarantees $\sigma_R > 0$ within each group.