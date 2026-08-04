# Skill candidates from the Fiji plugin ecosystem

A survey of the Fiji/ImageJ plugin ecosystem for procedures worth a biopb skill.
Source: the `fiji` GitHub org (~160 repos, roughly half archived/obsolete), the
major out-of-org plugins Fiji ships or links, and `imagej.net/list-of-extensions`.

**The bar** is [`write-a-skill`](../src/biopb_mcp/mcp/_skills_data/write-a-skill.md):
multi-step, with **decisions between the steps**, and standard practice in
bioimaging rather than in general Python. A plugin being popular is not the
signal — a popular plugin that wraps one call is still one call.

**A tier here is a hypothesis until it is measured.** The bar is about what a
model gets wrong *unaided*, which is not knowable by reading the plugin — it
takes `write-a-skill` step 6, a cold-context run scored against ground truth.
**Tier 1 is screened** (3 survived, 2 dropped, 1 exempt); **Tier 2 is screened
except the two procedural entries** (1 marginal, 4 dropped, 1 inconclusive);
**Tier 3 is not**. Screen before authoring, not after: of twelve candidates put
through it, **seven turned out to be things the model already does well** — and
in most of those the *procedure* I had written down as non-obvious was
reproduced unaided, sometimes better than my own reference.

**Out of scope for this survey**, by category:

| Excluded | Why | Examples |
|---|---|---|
| I/O, formats, readers/writers | Not a procedure | Bio-Formats, N5/HDF5, LSM/Vaa3d/H5J, simplified-io |
| Workflow / batch builders | The agent *is* the pipeline layer | JIPipe, MIA, BAR, ActionBar, HPC Workflow Manager |
| Viewers and figure tools | `guide://viewer` territory, not a skill | BigDataViewer, sciview, 3D Viewer, ClearVolume, FigureJ, Volume Viewer |
| Manual/interactive annotation | No procedure to describe | Cell Counter, Manual Tracking, ROI 1-click tools, Time Stamper |
| One-call operations | Explicitly excluded by `write-a-skill` | Auto Threshold, CLAHE, Hough Circle, NL-means, Gray Morphology, 3D Objects Counter |
| Obsolete / archived | — | script interpreters, `java6-*`, `javac`, Threshold_Colour, legacy-imglib1 |

## Already covered

| Fiji plugin | biopb skill |
|---|---|
| Grid/Collection Stitching (`fiji/Stitching`) | `stitch-tiles` — 2D grid only; see [T3](#tier-3--watch-list) for the multiview case |
| Correct 3D Drift (`fiji/Correct_3D_Drift`), StackReg/TurboReg, Image Stabilizer | `drift-correction` (pystackreg) |
| BaSiC-style shading correction | `flatfield` |
| Set Scale + Analyze Particles in calibrated units | `calibrated-measurements` |
| — (no Fiji equivalent; segmentation scoring is usually ad-hoc) | `segmentation-qc-metrics` |
| Rolling Ball Background Subtraction | kernel plugin `rolling_ball.py`, not a skill |
| ThunderSTORM (`zitmen/thunderstorm`) | `.claude/skills/smlm-localization.md` — repo-local, **promotion candidate** |
| AccPbFRET / RiFRET / FRETENATOR | `.claude/skills/molecular-tension-sensor.md` — repo-local and dataset-shaped; the general 3-cube case is [T2](#tier-2--worth-authoring-plugin-or-package-tier) |

## Tier 1 — author next

Clear procedure, clear decisions, a route that lands in the kernel with no new
heavyweight dependency, and a failure mode an unaided agent reliably hits.

**All four survived a Sonnet prescreen** (protocol at the end of
[rejected](#deliberately-rejected)): two cold arms per candidate, scored
against ground truth, with a reference implementation run on the same scenes to
prove the benchmark was winnable. A candidate that the cold arms *solved* was
dropped, not promoted.

| Candidate | Upstream | What is non-trivial | Route |
|---|---|---|---|
| **align-stack-by-features** | Linear Stack Alignment with SIFT (`axtimwalde/mpicbg`), Register Virtual Stack Slices | Complements `drift-correction`, which assumes small translations. The decision is *not* "reference frame vs. chained" — it is **direct-to-reference with a chained fallback gated on inlier count**, because inliers decay as section content drifts (measured: 84 → 24 → 10 over 4 sections) and the direct match dies silently when they run out | `skimage.feature.SIFT/ORB` + `measure.ransac`, inline |
| **count-foci-per-cell** | Find Maxima, SpotCounter, Foci Analyzer | Two-level segmentation: parent objects, then spots, then spot→parent assignment with the zero-count cells a naive join drops. The load-bearing part is that the detection threshold is applied to the **background-subtracted** residual — the same `median + k·MAD` rule on the raw channel silently returns nothing | `skimage.feature.peak_local_max` + label assignment, inline |
| **skeleton-network-metrics** | AnalyzeSkeleton + Skeletonize3D (`fiji/AnalyzeSkeleton`) | Skeleton → graph is where it goes wrong: spur pruning before counting, junction *clustering* (a thick junction is many degree-≥3 voxels, not many junctions), loops, and length accumulated per-axis in physical units | `skimage.morphology.skeletonize` + `pkg:skan` |
| **track-objects** | TrackMate (`trackmate-sc/TrackMate`) | LoG detector scale tied to expected radius (`σ = r/√2`), then linking cost vs. gap-closing vs. split/merge as *separate* budgets; frame-to-frame max distance derived from displacement statistics, not guessed | `pkg:laptrack` — **issue #669**, verification already done |

`track-objects` is the one Tier-1 entry **not** prescreened, and it is here on
different grounds: not that the model cannot derive the procedure, but that
`laptrack` already exists and a skill saves the iterations of wiring it up.
Judge it on that when it is written, not on an ablation.

## Tier 2 — worth authoring (plugin or package tier)

| Candidate | Upstream | What is non-trivial | Route |
|---|---|---|---|
| **pixel-classifier-segmentation** † | Trainable Weka Segmentation (`fiji/Trainable_Segmentation`), Labkit (`juglab/labkit-ui`) | **Elicitation and validation, not the classifier.** *Confirm-input:* the class list and what each one means; that the user will draw the scribbles, and how much is enough (every class present, boundary pixels included, more than one field); the feature scale range, tied to object size rather than left at the default. *Validation:* score on a **held-out** region — training-pixel accuracy is near-1 by construction and means nothing; report per-class balance and the fraction of pixels the classifier is unsure about; apply to a second field before declaring it done, since a model trained on one field commonly collapses on the next | `skimage.feature.multiscale_basic_features` + sklearn RF, plugin tier |
| **deconvolve-widefield** ⚠️ | Iterative Deconvolve 3D, Parallel Iterative Deconvolution, DeconvolutionLab2 | PSF provenance (measured beads vs. Gibson–Lanni from NA/λ/RI), iteration count as a bias/variance tradeoff, background handling — and the *When NOT to* that matters: do not deconvolve before intensity quantification. **Prescreen inconclusive** — see below; the one model-independent finding is that both arms validated by deconvolving with the PSF they had just generated, a circular check that cannot fail | `skimage.restoration.richardson_lucy` + `pkg:psfmodels` |

⚠️ **`deconvolve-widefield` is unresolved, not cleared.** Both cold arms built
defensible physics from NA/λ/RI — one a scalar angular-spectrum model, one
Born & Wolf — and scored NCC 0.48 and 0.66 against a 0.37 blurred baseline,
under a reference that scored 0.75. But that reference deconvolved with the
*exact* PSF used to generate the blur, so it is an oracle, and the arms'
shortfall measures PSF-model disagreement with the synthetic generator rather
than implementation quality. A valid test needs a truth PSF from a family no
arm would pick — a measured bead stack is the honest one — or it must supply
the PSF and score only the iteration/background half. Do not read the numbers
above as evidence either way.
| **landmark-registration** | BigWarp (`saalfeldlab/bigwarp`), bUnwarpJ (`fiji/bUnwarpJ`) | Cross-modality alignment from point correspondences: TPS vs. affine by landmark count, landmark placement strategy, and residual reporting. Fits napari well — a Points layer *is* the landmark UI | `scipy` TPS / `skimage.transform`, plugin tier |
| **ratiometric-fret** † | AccPbFRET, RiFRET, FRETENATOR | **Elicitation and validation, not the arithmetic.** *Confirm-input:* which channel is donor, acceptor and FRET; whether single-label control samples exist and were acquired under **identical** settings (without them the coefficients cannot be derived and the workflow stops here); the background region per channel; and whether the ratio is for display or for quantification, which changes what is allowed downstream. *Validation:* after correction a donor-only and an acceptor-only control must both ratio to ≈ 0 — that is the test that the coefficients are right; then check the ratio is **not** correlated with donor intensity, which is the signature of incomplete bleedthrough correction; report the ratio distribution inside the mask, never the ratio image alone | numpy, inline — generalizes the repo-local tension-sensor skill |

† **These two are kept on procedural grounds and the code prescreen does not
apply to them.** Their value is normalising *how the agent asks and checks*:
which facts to get from the user before computing (`write-a-skill`'s
confirm-input, spent from a budget of at most three blocking checkpoints), and
which validation actually discriminates. A cold-arm benchmark cannot see any of
that — it hands the arm a fixed signature and synthetic data with nobody to ask
and no control sample to check against, so it measures the one dimension these
candidates are not about. Judge them by dogfooding against a real user and real
controls, not by ablation.
| **detect-filaments** ‡ | Ridge Detection (`thorstenwagner/ij-ridgedetection`), FilamentDetector | Scale selection is fine unaided (both arms reached Lindeberg's rule). The measured failure is **thresholding the ridge response globally when filament brightness varies**: one arm's Otsu cut kept only the brightest filaments and scored recall **0.17 at precision 0.94** — and reported a mean width of 0.73 µm against a truth of 0.73, because the width was right on the sixth of the filaments it found. Nothing in the output says the rest are missing, so the validation rule is the skill: **check recall against the mask area, never precision alone** | Hessian eigen-analysis, plugin tier |

‡ **`detect-filaments` is marginal, kept rather than cleared.** One of two cold
arms matched the reference (F1 0.949 with a width prior, 0.910 without, against
0.943); the other scored 0.282. A single passing arm is not the clean Sonnet
pass that decides a rejection, and the losing arm's failure is invisible from
its own output. A second pair of arms would settle it.

## Tier 3 — watch list

Real procedures, but heavy, niche, or waiting on something.

- **BigStitcher / multiview-reconstruction** (`JaneliaSciComp/*`) — the part
  `stitch-tiles` lacks is *global optimization*: least-squares over pairwise
  shifts on a link graph instead of a spanning tree, plus outlier link rejection.
  Better as a version bump to `stitch-tiles` than a new skill.
- **SNT** (`morphonets/SNT`) — Sholl and Strahler analysis are well-defined and
  skill-shaped; the tracing UI is not.
- **CARE / Noise2Void / DenoiSeg** (`juglab/n2v`) — inference belongs on the
  algorithm plane and needs no skill, for the same reason deep-learning
  segmentation does not. Training is the only part left over, and that is
  environment management until the plane can host a fit.
- **Mastodon** (`mastodon-sc/mastodon`) — lineage tracking at a scale that wants
  its own storage story.
- **Kymograph** (`fiji/KymographBuilder`, Multi_Kymograph) — line-ROI resampling
  and velocity-from-slope; small, and a good napari Shapes-layer fit.
- **Squassh** (MOSAIC) — joint deconvolution + segmentation for subcellular
  objects; principled, and heavy.
- **NanoJ-SQUIRREL / eSRRF**, **FLIMJ**, **SIMcheck**, **LUMoS spectral
  unmixing** — each needs a data modality we cannot assume.
- **Focus quality** (`fiji/microscope-image-quality`) — the shipped model is
  TensorFlow, but a per-slice focus metric to find the best plane and flag
  defocused tiles is a small, useful skill.

## Deliberately rejected

Named here so the same candidates are not re-surveyed. **Two kinds, and they
have different futures:**

- **Structural rejects** — not a skill at any model tier, because the work is
  not the agent's. `segment-nuclei-dl` and CARE/N2V inference belong to the
  algorithm plane. These stay closed.
- **Tier-conditional rejects** — everything else below. They failed only
  because a Sonnet-class model already does them well, and the Haiku run
  (§ *Skill-worthiness is a property of the model tier*) shows the same
  procedures collapsing one tier down. **If lower-tier usability becomes a
  goal, these are the backlog** — already scoped, with the failure mode
  measured and a benchmark written, so authoring is mostly transcription.
  Until then they are **low priority**: writing them now buys nothing for the
  model that consumes the catalog today, and every shipped skill dilutes
  `find_skills` ranking for the ones that do earn their place.

  In rough order of how badly the low tier failed them: fibre orientation
  (89° error and a false alignment claim), local thickness (naive 2×EDT, −70 %),
  then the untested-at-low-tier remainder — splitting touching objects,
  colocalization, stain separation, photobleaching.

- **Splitting touching objects** — MorphoLibJ marker-controlled watershed,
  Adjustable Watershed. **Ablated and dropped**, 2026-08-03. Was Tier 1 on the
  claim that the classic wrong answer is watershed on the raw intensity, and
  that anisotropic spacing and seed suppression are where an agent slips. Three
  cold-context agents (2 Opus, 1 Sonnet), no skill and no repo access, were
  scored against a synthetic nearest-centre partition with
  `segmentation_qc.match_labels`, F1@0.5:

  | Scene | cold agents | skill-informed reference |
  |---|---|---|
  | 3D, 7:1 anisotropic | 0.96 / 1.00 / 0.96 | 0.96 |
  | 2D, 3x size range | 0.76 / 0.80 / 0.72 | 0.72 |
  | 3D isotropic (control) | 1.00 / 1.00 / 1.00 | 1.00 |

  Every arm matched or beat the reference. All three used
  `distance_transform_edt(..., sampling=spacing)`, marker-controlled watershed
  masked to the input, and a seed rule derived from the data; none used raw
  intensity as the primary surface, and two bounded an intensity term
  explicitly so that intensity alone cannot create a seed. The size-variation
  case caps near 0.8 for **every** arm including the reference, so the residual
  is a method limit, not a knowledge gap a skill closes.

- **Colocalization coefficients** — Coloc 2, JaCoP. **Prescreened and dropped**,
  2026-08-03. Was Tier 1 on the claim that Costes automatic thresholding,
  Manders above threshold, and a block-randomization significance test are the
  procedure an agent skips in favour of a bare Pearson. Both cold Sonnet arms
  produced all three, unprompted, and agreed with a reference Costes
  implementation to three significant figures — thresholds within 3 grey
  levels, identical M1/M2/r on all scenes. Both also self-tested and fixed real
  bugs before returning (a one-sided p-value that called an r = −0.99 exclusion
  pattern non-significant; a bisection that assumed a monotonicity the Costes
  threshold sweep does not have). Note also what *failed to be built*: no scene
  separated the correct procedure from the naive one — the intended "independent
  puncta" trap turned out to contain genuinely correlated shared
  autofluorescence, so a high r is the right answer there and reference and arm
  both report it. Being unable to construct the trap is itself the finding.

- **Deep-learning nuclei segmentation** — StarDist, Cellpose. Not a skill,
  because it is not a procedure the agent carries out: the model runs on the
  **algorithm plane** and the server owns tiling, fusing and the lazy data
  read. The agent hands over an `array_id` and gets one back. Tested
  previously; do not re-derive it as a `pkg:` skill on the grounds that
  environment and threshold management are hard — on this deployment they are
  not the agent's.

- **H&E stain separation** — Colour Deconvolution (`fiji/Colour_Deconvolution`).
  **Prescreened and dropped**, 2026-08-03. Both cold Sonnet arms produced
  Ruifrok & Johnston colour deconvolution — Beer-Lambert OD, unit-normalised
  stain vectors, cross-product third row, non-negativity clip — cited the
  paper, and cross-checked themselves against `skimage.color.rgb2hed`. Against
  known concentration maps: r = 1.00 (hematoxylin) and 0.98 (eosin), crosstalk
  ≤ 0.09, indistinguishable from the reference. It is one call plus a matrix,
  not a multi-step procedure with decisions between the steps.

  **Keep the one real finding, though**, because it belongs in a guide or a
  failure-modes row rather than being lost: the white point is *not* 255. On a
  scanner whose blank field reads ~235, assuming 255 puts a constant OD floor
  under every pixel. Correlation cannot see it — r stays 1.00 either way — and
  it shows up only as stain leaking into genuinely blank tissue, measured here
  at 0.06 (white estimated from a bright percentile) vs. 0.32 (assumed 255).
  One arm assumed 255 and reproduced the biased number exactly; the other
  half-handled it at 0.15.

- **Photobleaching correction** — CorrectBleach (`fiji/CorrectBleach`).
  **Prescreened and dropped**, 2026-08-03. The claimed non-trivial part was
  choosing between ratio / exponential fit / histogram matching according to
  what is being measured. **That claim did not survive contact**: both cold
  arms picked an exponential-plus-offset fit unprompted, with data-derived
  initial guesses and a monotonic-envelope fallback, and both matched the
  reference exactly on a pure-bleach series (1.00 vs. truth 1.00) and on a
  series with real biology (1.14, as did the reference). Measured on the scene
  that was supposed to separate the methods, ratio normalisation scored 1.24
  and the exponential fit 1.14 against a truth of 1.40 — the method choice
  barely matters.

  **Two findings to keep**, neither of them a skill:
  - *Subtract the camera offset before a multiplicative correction.* This is
    the one place the arms diverged: one subtracted a per-frame background
    median first and scored 1.01 against truth 1.00; the other rescaled whole
    frames, offset included, and scored **0.59** — a 41 % error in the reported
    trend, from a stack that still looks correctly flattened.
  - *Bleaching and a real global signal change are not separable from one
    trace.* When every object carried the same 40 % rise, no method recovered
    it — reference included (0.81 and 1.00 against truth 1.40). It becomes
    recoverable only when the responders are a minority, so the decay is
    estimable from the static remainder. Worth saying out loud somewhere,
    because it is a property of the experiment, not of the code.

- **Local thickness** — BoneJ2, LocalThickness. **Prescreened and dropped**,
  2026-08-03. The claim was that the naive "distance transform × 2" answer is
  wrong and the real definition has no clean Python equivalent. Half of that
  holds — naive 2×EDT reads 1.50 / 1.44 / 1.60 against a truth of 5.00 / 3.00 /
  2.40 on a sphere, rod and slab — but **no arm gave the naive answer**. Both
  named Hildebrand & Rüegsegger (the largest inscribed sphere *containing* the
  voxel), passed `sampling=spacing` for the anisotropy, and swept radii
  largest-first: 4.61 / 2.68 / 2.38 and 4.61 / 2.68 / 2.40, matching or beating
  the reference's 4.61 / 2.55 / 2.36. One validated itself against an
  anisotropic sphere at `spacing=(1,1,4)` and recovered 24.08 vs. 24 expected.
  A dumbbell — the case convex primitives cannot test, where a lobe's sphere
  could leak across a neck — did not separate them either: lobe 4.00 for every
  implementation including the reference (truth 4.40), neck 0.74–0.80 (truth
  1.10, voxel quantisation, and the reference takes the same hit). *The entry's
  premise was also confused:* "no clean Python equivalent" is a claim about
  library availability, which is true and irrelevant — an absent library
  function is exactly the case a model writes out from the definition.

- **Fibre orientation** — Directionality, OrientationJ. **Prescreened and
  dropped**, 2026-08-03. The claim was circular statistics — that a mean of
  angles is not a mean. Both arms built a global structure tensor, whose
  double-angle form handles the wrap *by construction*, and both scored 0.1°
  and 0.7° error on a 30° population and on a population straddling the 180°
  wrap (175°/5°, where an arithmetic mean would return 90°), beating the
  reference's 0.7°/0.7°. Both also returned coherence ≈ 0.05 on an isotropic
  negative control, and both independently flagged that a fibre is undirected
  so the answer wraps mod 180°.

**A pattern across the rejects.** `separate-stains` and
`correct-photobleaching` failed the same way: a procedure the model already
knows cold, carrying exactly one caveat that decides whether the numbers are
right (the white point; the camera offset). A caveat is a `guide://` line or a
*Failure modes* row in a neighbouring skill — it is not a skill file, because
there are no decisions between the steps for it to sit between. Check a
candidate against this shape before benchmarking it: if the procedure collapses
to one call plus one warning, the survey has already answered.

### Skill-worthiness is a property of the model tier

Every rejection here is conditional on the consuming model, and the size of
that condition was measured, not guessed. Three rejected candidates were re-run
against **Haiku**, which failed all of them — at exactly the traps the entries
had predicted:

| | Haiku | Sonnet | truth |
|---|---|---|---|
| local thickness (sphere/rod/slab) | **1.50 / 1.44 / 1.60** — bit-identical to naive 2×EDT | 4.61 / 2.68 / 2.40 | 5.00 / 3.00 / 2.40 |
| orientation, single population | err **29.7°** (no gradient→fibre 90° rotation) | err 0.1° | 30° |
| orientation, 175°/5° wraparound | err **89.1°** — the arithmetic mean of angles | err 0.7° | 0° |
| coherence, isotropic control | **0.82** — "highly aligned" for random fibres | 0.05 | ≈ 0 |
| filaments | did not run: called a `hessian_matrix_eigvals` signature removed from skimage, and never executed its own code | F1 0.91–0.95 | — |

So the entries were not wrong about the failure modes — they were wrong about
*whose*. **A catalog written for a cheap tier would keep nearly everything
rejected on this page.** Two consequences:

1. These rejections hold only while skills are consumed by a Sonnet-class model
   or better. Pointing the catalog at a cheaper tier invalidates them silently,
   and they would need re-running rather than trusting.
2. Prescreening at Sonnet is right not merely because Opus is stronger, but
   because the tier gap is where the whole signal lives. Screening at the same
   tier that consumes the skill would measure nothing.

### The prescreen protocol

What the runs above converged on, for the next candidate:

1. **Two cold arms per candidate**, fresh context, no skill, no repo, no web,
   varied scenario framing but an **identical fixed signature** so the arms are
   directly scoreable against each other.
2. **Sonnet is the bar.** A Sonnet pass is decisive for rejection, since the
   Opus-class model that would consume the skill is stronger. A Sonnet failure
   is inconclusive and escalates to a full Opus ablation.
3. **Commit the skill-informed reference before looking at any arm**, so it
   cannot be tuned to the benchmark.
4. **Prove the benchmark is winnable.** Run the reference on the same scenes
   first. Two candidates here were nearly misread as arm failures until the
   reference showed the scene *was* recoverable (foci: ceiling 21/25) or that
   the arm simply equalled it (coloc). An unwinnable scene scores every arm
   zero and teaches nothing.
5. **Probe each claimed-load-bearing parameter by feeding a false value** —
   passing `spacing=(1,1,1)` for anisotropic data separates real use from
   decorative use.
6. **Keep the arms' scenario prompts equivalent.** The skeleton run failed this:
   one framing mentioned ragged masks and the other did not, so the two arms
   were not comparable. Salvageable as a finding, but do not plan for it.

- **Auto Threshold / Auto Local Threshold** (`fiji/Auto_Threshold`, 45★) — the
  16 methods are `skimage.filters.try_all_threshold`. `write-a-skill` excludes
  named auto-thresholds by name.
- **3D Objects Counter**, **Analyze Particles** — label + `regionprops`, and the
  calibrated part is already `calibrated-measurements`.
- **Statistical Region Merging**, **SIOX**, **level sets**, **Balloon
  Segmentation**, **Graph Cut** — superseded in practice by the Tier 1/2
  segmentation routes.
- **Kuwahara / anisotropic diffusion / wavelet & stochastic denoise** — one call
  each in `skimage.restoration` / `skimage.filters`.
- **Temporal-Color Code**, **Glasbey**, **Simulate Color Blindness** — display.
