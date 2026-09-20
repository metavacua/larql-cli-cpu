# V3-LENS-1 — can the executor's own head read an intermediate carrier faithfully, and at a declared price?

Pre-registered 2026-09-20, before any implementation. Properties and forecasts below are
FROZEN. Nothing here claims a result.

Programme: OBSERVE. Rung 3, above V3-OBS-1 (#484) and V3-STREAM-1 (#485), beside the
`observe` verb (#486). The first Anatomist-style research lens: what the model would say
at each depth, computed by the model's own head and nothing else.

---

## The question

V3-OBS-1 deliberately refused a logit lens. Its selected-token probe is a raw dot product
of the carrier against supplied rows, with no final norm, no head multiplier, no softcap,
and it says so. That was right for a cheap observation statistic, and it is not what a
researcher means by "P(Paris) at layer 24".

A **true** logit lens applies, to the carrier at an intermediate depth, exactly the
operations the executor applies at the exit: the prepared final norm and the prepared
output head, with the head's multiplier and softcap, on the head's real weights (tied
or not, in whatever representation the image pinned). Anything less is a different
reader with a different name.

V3-LENS-1 asks whether that can be done **through one code path** — the exit's own — so
that the lens at the last layer *is* the executor's logits rather than agreeing with
them, and whether its cost, one head pass per tapped site, can be declared and measured
rather than hidden inside "observation".

---

## What already exists (read before implementing)

| Fact | Where |
|---|---|
| The exit: `final_hidden = final_norm.apply(backend, exit_hidden)` then `backend.output_head(weight.slice(), vocab, hidden, &final_hidden, op.multiplier, op.softcapping)` | `opplan/exec/decode.rs`, the `logits =` match at the end of `run` |
| The same two operations on the batch path | `opplan/exec/mod.rs` |
| `PreparedOperands::final_norm()` and `::output()` are `pub(super)` — reachable inside `exec`, not by a consumer | `opplan/exec/prepared.rs` |
| The carrier at every write, borrowed, with the layer scale where the program has one | `carrier_write` tap (V3-OBS-1); the layer output is `layer_scale × after` (C5) |
| A consumer that records events with a run-scoped sequence and a receipt | `RunRecorder` (V3-STREAM-1) |
| The head's cost class in the timing ledger | `opplan/exec/timing.rs` |

---

## Contract properties (frozen as properties, not names)

**H1 — one head path.** The executor exposes one function on the prepared image that
takes a `[hidden]` carrier and a backend and returns the head's logits by applying the
prepared final norm and the prepared output head. The exit calls that same function.
There is no second implementation of "the head" anywhere.

**H2 — the lens reads the layer output, not the write.** On a site whose program applies
a layer scale after the write, the lens reads `layer_scale × after`, because that is
what the next layer and the exit would read. The lens says which site it read.

**H3 — the lens is a consumer.** It is a `StepObserver` that, at the sites it was
declared for, copies the carrier and calls the head function. It never changes
execution; V3-OBS-1's parity gate covers it like any observer.

**H4 — sites are declared, the price is per site.** A lens is armed with an explicit
site set: every layer, every k-th layer, or a list; attention site, FFN site or both.
Its cost is one head pass per armed site per token, and the record says how many.

**H5 — outputs are the distribution's facts, not a probe's.** Per armed site and per
declared token, the lens records the log-probability under the full log-softmax and the
rank; optionally the top-k ids. Full logits are never persisted by this rung.

**H6 — the head is the image's head.** Whatever representation the image pinned for the
output head (bf16, f32, Q8 under the default policy) is what the lens computes with, and
the run's provenance already names it. The lens does not carry a head of its own.

---

## Frozen acceptance properties

**LP1 — parity.** Logits at every position with the lens armed on every site are
bit-identical to the unobserved run, on the reference and production CPU backends.

**LP2 — the anchor.** At the last layer's FFN site, the lens's logits for a position are
**bit-identical** to the executor's own logits for that position, on the golden plan and
on every real subject run. This is H1 made observable: the same function on the same
vector.

**LP3 — the exit is the function.** The exit's code path is the shared function (a
refactor, witnessed by the existing decode-vs-batch and observed-vs-unobserved gates
staying bit-identical before and after).

**LP4 — a proper distribution at every depth.** At every armed site the full
log-softmax is finite and the declared tokens' log-probabilities are at most zero and
sum, with the rest, to one within f64 rounding.

**LP5 — cost is measured, not described.** Token wall with the lens armed on every FFN
site minus token wall with `NoopObserver`, on Granite 4.2 3B and on Gemma 3 4B IT,
release, production CPU; reported per token and per armed site. The forecast below
says why Gemma's number will be large.

**LP6 — the lens on the record.** A lens readout is a recorded event with its sequence,
site, and per-token log-probability and rank; replay is equal; the receipt covers it.

**LP7 — the first real reading.** Gemma 3 4B IT, `The capital of France is`, the lens
armed on every FFN site for the tokens ` Paris` and ` France` at the final prompt
position: the record carries their log-probability and rank at every layer. **No shape
is forecast**: the ADDRESS-BUILD finding (relation early, entity late) is a prior about
Gemma 3 and is what this reading will be compared against, not assumed.

---

## Forecasts

- **LF1**: LP2 holds bit-for-bit on the golden plan (both backends), Granite and Gemma 3
  4B (production).
- **LF2**: cost per armed site is one output-head pass. On Gemma 3 4B that is a
  262,208 × 2,560 projection, roughly 0.67 GMAC per site; armed on all 34 FFN sites
  that is about 23 GMAC per token against a forward of a few GMAC, so a full-depth lens
  on Gemma is expected to cost **several times** the forward it observes. That is why H4
  exists. Granite's 49,152 × 2,560 head is about 5× cheaper per site.
- **LF3**: no change to the V3-OBS-1 observer contract or the STREAM-1 record schema
  beyond one new event kind; if either needs more, it is recorded as a finding.
- **LF4**: the CLI gains `--lens-tokens`, `--lens-sites` and nothing else.

---

## Out of scope

Per-head attention observation (its own rung), interventions, tuned or learned lenses
(a "tuned lens" is a different reader and would be named as one), the Metal path, any
lens on the batch prefill path.

---

## Verdict rule

V3-LENS-1 is complete only when LP1–LP6 PASS and LP7 is recorded. It unlocks the
Observatory's Logits tab and the depth-wise answer trajectory, both reading the record.
