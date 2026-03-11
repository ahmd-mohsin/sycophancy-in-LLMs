Now in the methodlogy section after the disentagled reward, i have this two pahse training. Alot of the stuff is already written before in methodology . Remove that, also the stuff int he evalaution secti:

\subsection{Two-Phase Training Pipeline}
\label{sec:method_training}

\paragraph{Phase 1 — SFT warmup.}
Before RL training, we fine-tune $\pi_{\mathrm{ref}}$ on pressure-free
baseline responses via supervised cross-entropy:
\begin{equation}
  \label{eq:sft}
  \mathcal{L}_{\mathrm{SFT}}(\theta) \;=\;
  -\mathbb{E}_{(c,q) \sim \mathcal{D}} \Bigl[
    \log \pi_\theta\!\bigl(r_\emptyset(c,q) \;\big|\; \emptyset, c, q\bigr)
  \Bigr],
\end{equation}
optimised over both context orientations. This serves two purposes: (i) it
establishes a prior $\pi_{\mathrm{ref}}$ that already knows what a
pressure-free, evidence-grounded response looks like, making the KL penalty in
Eq.~\eqref{eq:grpo_obj} meaningful; and (ii) it prevents the GRPO policy
gradient from spending capacity on learning the basic task format, which would
compete with learning pressure resistance.

\paragraph{Phase 2 — GRPO with disentangled reward.}
Starting from the SFT checkpoint, we optimise Eq.~\eqref{eq:grpo_obj} with
$R$ instantiated as Eq.~\eqref{eq:reward_total}. For each prompt $(P_j, c, q)$
in the training set, $G=4$ completions are sampled, rewarded, and
group-normalised per Eq.~\eqref{eq:grpo_adv}. The KL coefficient is set to
$\beta = 0.2$, calibrated to prevent the policy collapse observed in
preliminary runs where $\beta = 0.04$ led to KL divergence above $0.74$ and
catastrophic regression on PACF.

\paragraph{Prompt construction.}
Each training prompt is assembled as:
\begin{equation*}
  x \;=\; [\texttt{SYSTEM}]\; s \;\;
           [\texttt{USER}]\; P_j \;\|\; \texttt{Context: } c \;\|\;
           \texttt{Question: } q,
\end{equation*}
where $s$ is a fixed system prompt instructing the model not to allow authority
or credentials to influence its answer, and $\|$ denotes string concatenation.
An index tag \texttt{\_\_GRPO\_IDX\_}$k$\texttt{\_\_} is prepended to each
prompt so the live reward function can recover the group metadata
$(b(C), b(C'), o, \mathrm{category})$ required to evaluate
Eqs.~\eqref{eq:rp}--\eqref{eq:rg} without passing auxiliary state through the
trainer interface.

\subsection{NLI-Based Semantic Distance}
\label{sec:prelim_nli}

Definitions~\ref{def:pi}--\ref{def:sycophancy} require a semantic distance
$d(\cdot,\cdot)$ over responses. We instantiate this using a cross-encoder
NLI model $f_\phi$~\citep{he2021deberta} that maps a (premise, hypothesis)
pair to a simplex over $\{\texttt{entailment}, \texttt{neutral},
\texttt{contradiction}\}$. For two responses $a, b \in \mathcal{Y}$:
\begin{equation}
  \label{eq:shift}
  d_\phi(a,b) \;=\; \text{shift}(a,b) \;\coloneqq\;
  \frac{p_{\mathrm{contra}}(a,b)}
       {p_{\mathrm{entail}}(a,b) + p_{\mathrm{neutral}}(a,b)
        + p_{\mathrm{contra}}(a,b)},
\end{equation}
where $p_{\mathrm{contra}}(a,b) = [f_\phi(a,b)]_{\texttt{contradiction}}$
and similarly for the other classes. Normalising over all three classes is
necessary: a hedged response that neither entails nor contradicts scores
$p_{\mathrm{contra}} \approx 0$ and would appear falsely pressure-resistant
under a two-class ratio. The evaluation metrics in Section~\ref{sec:metrics}
are defined in terms of $\text{shift}$ and the complementary entailment score
$p_{\mathrm{entail}}(a,b)$ directly.

\paragraph{Why NLI over LLM-as-judge.}
A sycophantic LLM judge assigns higher scores to agreeable
responses~\citep{wang2023large}, creating circular validation. DeBERTa
cross-encoders are discriminative classifiers trained on sentence-level
entailment (SNLI/MultiNLI~\citep{bowman2015snli,williams2018multinli}) and
carry no social-agreement prior.

\subsection{The Reward Decomposition Principle}
\label{sec:prelim_decomp_reward}

A scalar reward $R(x,y)$ conflates the two orthogonal failure modes in
Eq.~\eqref{eq:syco}. To see why this is problematic, consider a response $y$
that is factually correct but phrased in a pressure-resistant tone: it
satisfies pressure independence but may receive a low scalar reward if the
reward model encodes an agreement bias. Conversely, a perfectly agreeable but
context-faithful response receives high reward despite violating pressure
independence.

We therefore replace the scalar $R$ with a \emph{disentangled} reward vector:
\begin{equation}
  \label{eq:decomp}
  R(x, y) \;=\; f\bigl(R_p(x,y),\; R_c(x,y),\; R_{\mathrm{pos}}(x,y),\;
  R_g(x,y),\; R_q(x,y)\bigr),
\end{equation}
where each component is computed via NLI against a context-matched reference
(the pressure-free baseline $r_\emptyset$), and $f$ is a weighted linear
aggregation. The key property is \emph{orthogonality}: each component targets
exactly one of the two independence conditions in
Definitions~\ref{def:pi}--\ref{def:er}, and no two components are
simultaneously maximised by the same failure mode.

Specifically:
\begin{itemize}[leftmargin=*, noitemsep]
  \item $R_p$ and $R_{\mathrm{pos}}$ enforce Pressure Independence
        (Definition~\ref{def:pi}) — they are maximised when the response
        is invariant to $P$.
  \item $R_c$ and $R_q$ enforce Evidence Responsiveness
        (Definition~\ref{def:er}) — they are maximised when the response
        entails the context-appropriate reference.
  \item $R_g$ penalises the degenerate equilibrium in which the policy
        satisfies neither condition by outputting hedged, content-free
        agreements.
\end{itemize}
The full specification of each component, their NLI instantiation, and the
weighted aggregation formula are given in Section~\ref{sec:methodology}.on, is already present so remvoe stuff from thes section which repeats itself later. Remove verbosity and make it proper and breif and very research oriented. Done use any --, and also dont keep any undefied acronyms