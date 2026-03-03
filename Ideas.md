Here is some issue thta are spotted by cursor. We are using older datasets.

From your inspect_rewards output there are two clear points and one thing to watch.
1. Stored vs live totals (expected)
Stored totals (e.g. 0.9257, 0.7098) are lower than live totals (e.g. 1.2133, 1.1886) because:
Stored reward_dataset_20260227_012127.json was built with the v2 formula: no Rpos, and delta=0.3.
Live recomputation uses current code: Rpos (ε·Rpos) is included and default delta=0.6.
So the reward dataset is from an older formula. Totals are not supposed to match; the dataset itself is consistent with that older spec.
2. Rp mismatch on “opposite” samples (real inconsistency)
For the same sample, stored and live rp often disagree, especially for opposite context:
Sample	Stored rp	Live rp
political_0000_low_opposite	0.481	0.608
political_0000_medium_opposite	0.481	0.608
political_0000_high_opposite	0.481	0.608
political_0037_low_opposite	0.280	0.470
political_0037_medium_opposite	0.280	0.470
political_0037_high_opposite	0.280	0.470
So:
Stored: same rp (e.g. 0.481 or 0.280) for all three pressure levels within “opposite” for a topic.
Live: different rp, and often the same value (e.g. 0.608 or 0.470) for low/medium/high opposite.
So either:
When the dataset was built, the code that wrote rp for (level, opposite) was wrong (e.g. reusing one value for all three levels, or using the wrong NLI pair), or
NLI is non-deterministic (e.g. device, bfloat16, batching) so the same (baseline_orig, response) pair gives different shift scores then vs now.
Rq, rc, and rg match in the output; the only component that systematically disagrees is rp on opposite-context samples.
3. What this means for the “dataset”
Structure and counts are correct (395 groups, 2370 samples, pressure/context balance).
NLI-based components (rq, rc, rg) recompute and match the stored values in the check you ran.
Rp for opposite-context is inconsistent between stored and live; that’s the only dataset/reward bug or nondeterminism visible here.
4. Recommendation before GRPO v4
Rebuild the reward dataset with the current reward formula (including Rpos and your chosen delta) so stored totals and components match what the trainer will use.
Fix NLI determinism for reward building: same device, model.eval(), and ideally same dtype (e.g. float32) when calling nli_score_batch so that re-runs give the same rp (and thus same advantages).
Optionally add a small script that recomputes rewards for a few groups and asserts stored rp == live rp (and optionally other components) within a small tolerance; run it after building the reward dataset to catch bugs or variance.
So: the “issue” in the dataset is that (i) it was built with an old reward formula (hence total mismatch), and (ii) rp for opposite-context samples is inconsistent with a fresh NLI run, which points to a bug in how rp was written for opposite context and/or NLI non-determinism.


Now moving forward with, I have attached the files you require.
