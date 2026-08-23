# Tier E: exploratory ablations

**Not a preregistration file. Nothing in here is a result.** Tiers 1–3 make claims, here we don't. doesn't. It's purpose is purely to try things. Lettered rather than numbered because it isn't the next rung, rather it sits beside the sequence, not in it.

Written 23-08-2026 @2:150pm before the Tier 2 seal was opened. 

## Rules

- **Validation individuals only. Test partitions are out of scope.** That is what keeps Tier 3 confirmatory; needs those partitions unread.
- No success bands, no pass/fail. Everything here is descriptive.
- Registered constants can be broken freely. That's the point. Each variant records which ones.
- A Tier E finding isn't a claim on its own; it hasn't been tested on anything it didn't help choose. So if something here works, we will register it and test it. Then it can be the headline.
- It doesn't gate or delay Tier 3.

We only use validation data; since that is what it is for. Then something here can legitimately feed into a later registered arm without contaminating it, because no test data went into finding it.

## Queued up

Rough order of what I want to do (and hence what made me want to add this file to the repo):

1. **Representation fixes.** for the low displacement collapse Tier 2 measured; winsorise the normalised inputs, raise $\sigma$, floor at a percentile, pre LN trunk. 
2. **Batch size.** So far our Tier 2 has used 12% of an H200 at batch 256 over 13 tokens.
3. **Pretraining variants.** Two stage, and causal next step. 
4. **Environmental covariates.** Worth trying once the representation works. Note that we only have a small source pool size, a domain specific covariate is close to a domain indicator; the same collinearity problem which is why we pushed trait conditioning out of Tier 2.
5. **Trait conditioning**, descriptively.

Not exhaustive; I'll keep adding things as I go and this file is timestamped unlike my ideas in `notes.md` (gitignored) so it is legit. 
