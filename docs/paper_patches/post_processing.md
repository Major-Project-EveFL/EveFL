# Paper patch: key post-processing and the finite-key bound (Phase 1, step 3)

Labels: **simulated** = produced by the pipeline in `evefl/quantum/postprocess.py` on simulated exchanges
(`docs/validation/key_rates.json`); **analytic** = evaluated formula. The keys are a deterministic function of the
simulation seed, so they are NOT secret; this is a pipeline and key-length accounting, not real QKD security.
Not wired into the FL path yet (Phase 4).

## What is implemented
Parameter estimation with the compared bits discarded from the key, Cascade error correction with every disclosed
parity counted, a verification hash of ceil(log2(1/eps_cor)) bits, privacy amplification by random Toeplitz hashing
to a length given by the finite-key bound, and authentication of the classical transcript. Hash seeds and the
authentication key come from `secrets` / `os.urandom`. Authentication is HMAC-SHA256 (computational), not
Wegman-Carter, and must be described that way.

## The bound (verified against the paper)
Tomamichel, Lim, Gisin, Renner, Nat. Commun. 3, 634 (2012), arXiv:1103.4130, Supplementary Theorem 2:
```latex
\ell \le \Big\lfloor n\big(1-h(Q+\mu(\varepsilon))\big) - 2\log_2\tfrac{1}{2\bar\varepsilon}
        - \mathrm{leak}_{EC} - \log_2\tfrac{2}{\varepsilon_{cor}} \Big\rfloor,\qquad
\mu(\varepsilon)=\sqrt{\tfrac{n+k}{nk}\,\tfrac{k+1}{k}\,\ln\tfrac1\varepsilon},\quad 2\varepsilon+\bar\varepsilon\le\varepsilon_{sec},
```
maximised over the split of epsilon; q = 1 for ideal single-photon BB84 and h is the binary entropy truncated at 1
above 1/2. We use eps_sec = 1e-10, eps_cor = 1e-15 and the OBSERVED sample error rate in place of the abort
threshold Q_tol (the paper's Lemma 3 gives Q_key < lambda + mu for the observed lambda, except with probability eps).

State these assumptions wherever the bound is used: (i) the paper estimates the phase error from a sample in the
complementary basis, whereas here the sample is a uniformly random subset of the sifted key; this is justified only
because the simulated channels (intercept-resend with random basis, bit flips) are basis-symmetric, so the result is
the bound's form for these channels, not a general-attack security proof; (ii) single-photon sources, no
detector side channels (blinding), simulation only.

## Result: block size needed for any key (simulated, sample fraction 0.1)
| channel | smallest tested block with a key | key bits per qubit at 2^17 | at 2^20 |
|---|---|---|---|
| no Eve, 1% bit flips | 2^13 = 8,192 qubits | 0.236 | 0.312 |
| alpha = 0.1, no noise | 2^14 = 16,384 qubits | 0.161 | 0.236 |
| alpha = 0.2, 1% bit flips | 2^17 = 131,072 qubits | 0.021 | 0.083 |

- **The headline controller setting (1024 qubits, 128-bit QBER sample) yields NO key**: the finite-key penalty exceeds
  the key at any error rate (negative bound even at zero errors). Key generation needs blocks of at least about 10^4
  qubits; the controller and the key pipeline may use different block sizes, which is a Phase 4 design decision.
- Cascade efficiency (leaked bits over n h(Q)) is about 1.3 to 1.4, above the 1.1 assumed in the paper's own
  optimisation; a better reconciliation protocol would raise the rates.
- With the key bounded by Q + mu, a weak Eve who is missed by the controller (for example alpha = 0.1, caught only 40%
  of rounds at 1024 qubits) does not compromise the key: in simulation the key length stayed below the entropy an
  intercept-resend Eve actually left (`tests/test_postprocess.py`, using the simulator's ground truth of which bits she
  knows). At alpha >= 0.3 with 2^17 qubits there is no key, and at alpha = 0.5 the round aborts (QBER above 0.11).
- Rounds without a key are treated like LOCKDOWN by the orchestration policy (`evefl/orchestration/policy.py`), with the
  separate reason `no_key`; the pure controller stays unaware of key length.
