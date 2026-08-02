---
id: write-a-skill
title: Write a new biopb skill file
description: Turn a workflow the user has just validated into a reviewed skill file for the biopb catalog.
tags: [workflow, authoring]
version: 1.1.0
requires: []
---

# Write a new biopb skill file

## When to use

The user has finished a workflow worth repeating and wants it captured, or asks
directly for a new skill. A skill is a markdown recipe retrieved by
`find_skills`, and it can live in either of two places: the user's own
`~/.config/biopb/skills/`, available to them immediately, or the curated catalog
published on biopb.org, which a maintainer reviews. Same file either way — the
destination is the user's choice, made at the end (step 7).

A skill is worth writing when the procedure is **multi-step**, has **decisions
between the steps**, and is **standard practice in bioimaging rather than in
general Python** — the recipe an agent would otherwise improvise wrongly.

## When NOT to use

- **The workflow is one call.** Filters, named auto-thresholds, projections,
  binary morphology, connected components, channel splits. An agent writes these
  correctly without help; cataloguing them dilutes `find_skills` ranking.
- **It is API reference.** How to add a layer, query the catalog, or run a job
  belongs in the `guide://*` resources, not a skill. A skill is judgment,
  parameter rules, and validation.
- **The run has not been validated.** Never author from a workflow the user has
  not confirmed on real data. A skill is a claim that the procedure works; an
  unverified one teaches every future session the same mistake.
- **It is specific to one dataset.** Particular file names, hard-coded crops, or
  one experiment's channel order belong in the conversation, not the catalog.

## Parameters

The frontmatter fields, and how to determine each:

| Field | How to determine it |
|---|---|
| `id` | Kebab-case, must equal the filename stem. Name the *task*, not the tool: `track-objects`, not `run-laptrack` |
| `title` | One line, imperative. What the user gets |
| `description` | One sentence. This is what `find_skills` ranks on — write it as the user's request, not as an implementation summary |
| `tags` | A list of categories describing the skill. Reuse tags you have seen on published skills where they fit — consistent tags are what make discovery work |
| `version` | `1.0.0` for a new skill. Bump on every content edit; the site derives `updated` from git, so never set a date by hand |
| `suggests` | Same vocabulary as `requires`, for what only the **preferred** path needs — a package the skill is better with and honest without. Use it when your body names a real fallback, and then step 1 must name the package *and* say what happens without it. This is what lets the skill ship to a platform the package has no wheel for: a required package missing there rejects the skill outright, a suggested one is a hole the degraded path covers. The workspace floor (`pkg:biopb-mcp>=X`) may never be suggested — there is no fallback from an interface that does not exist yet — and no token may appear in both lists |
| `requires` | What the steps actually touch: `viewer`, `tensor`, `dask`, `ops:<kind>`, `plugin:<name>` (the plugin's **file stem**, e.g. `plugin:rolling_ball` ↔ `rolling_ball.py` — also the name it is bound under in the kernel namespace, so the body calls `rolling_ball.subtract_background(...)`), `pkg:<name>` with a version bound — `~=X.Y.Z` for a third party (a floor *and* an upper bound at the next minor, so the API the body quotes cannot move under it; never a bare floor, never `==`, and never a comma pair, which the runtime reader splits), `>=X.Y.Z` only for `biopb-mcp` itself. Not decoration — the agent resolves this list against the live session before starting (see step 2). A skill that drives the kernel carries `pkg:biopb-mcp>=0.13.0`, the first release exposing the interface it is written against; raise that bound only if the skill needs something newer. An empty list is a real answer, not an omission: a skill whose steps touch nothing in the session has nothing to resolve |

## Steps

1. **Confirm the scope with the user** *(blocking)*. State in one sentence what
   the skill will cover and what it deliberately leaves out. First run
   `find_skills` for overlap — if an existing skill covers most of it, **edit
   that skill and bump its version** instead of adding a near-duplicate.

2. **Choose how the code ships.** This is the main scoping decision:

   | Amount | Ships as | The body carries |
   |---|---|---|
   | ≲ 30 lines | Inline code fences | The code itself |
   | 30–150 lines | A kernel plugin (`biopb_mcp/plugins/`, seeded to `~/.config/biopb/kernel/`) | The call signature — qualified by the plugin's module name — and what the parameters mean |
   | A published algorithm | A `pip install` pointer | The install command and the degraded path when it is absent |

   **Do not restate the install mechanics.** Which command to quote, why never a
   bare `pip install`, and what a managed environment does to an added package
   are all in `guide://kernel`, which ships with the server that prints them — a
   copy in your body is one more thing to go stale. Write the half that is yours:
   *which* packages, and *what the degraded path is*. That path is the load-
   bearing half of a package-tier skill anyway, since it is the one that survives
   an upgrade.

   **Do not write data-access code either.** Getting an array out of a layer or
   off the tensor server is `guide://data`'s job — pyramids, laziness, a `scale`
   that is per-axis rather than positional, what an upload drops. A snippet in
   your body is a
   second copy of mechanics that change with the loader, and a wrong one is the
   most expensive kind of wrong here: it runs, and the numbers are silently off.
   Name what your steps need from the data — *both label arrays at the same
   level*, *the spacing that belongs to this array* — and point at the guide for
   how. Code fences in a skill are for the skill's **own** algorithm and calls.

   Whatever the tier, **step 1 of the workflow is the requirement check**, before
   the confirm-input step — there is no point asking the user which layer is truth
   if the scorer was never going to be there.

   Keep *the check itself* to one sentence: resolve `requires:` (and `suggests:`,
   if you declared one) against `server_status`, and point at `guide://kernel` for
   what to do about a gap. A suggested package is the exception to the one-sentence
   rule: say in the same step what the run does without it, because that is the
   version of the skill some platforms only ever get. The
   diagnosis — which section answers which token, what a missing plugin means, the
   options for a missing package — lives in the guide, and a copy in your body is
   one more thing to go stale. Add a clause only where your own token needs
   pointing at ("the scorer is a kernel plugin, so `## Kernel plugins` answers for
   it").

   Then, in the same step, write the half the guide cannot know: the **degraded
   path** for each optional piece and what taking it costs the result — or that
   there is none, which is just as useful to say. It belongs beside the check,
   where the agent decides whether to go on.

3. **Draft the six required sections**: *When to use*, *When NOT to use*,
   *Parameters*, *Steps*, *Failure modes*, *Next steps*. Two carry most of the
   value and are the ones most often skipped:

   - **When NOT to use** — the negative case an agent cannot infer (do not apply
     background subtraction to a ratiometric image; do not deconvolve before
     quantifying).
   - **Failure modes** — a symptom → cause → fix table. An agent cannot debug an
     unfamiliar pipeline from first principles, but it can match a signature.

   Give parameters as a table of name, unit, and *how to derive the value from
   the data*. `radius = 1.5x the largest object diameter` is usable; `radius=50`
   is not.

4. **Put checkpoints in the steps, by name.** "Check with the user when
   appropriate" produces either constant interruption or none at all:

   - **Confirm-input** *(blocking, before compute)* — only for facts not
     derivable from the data: voxel spacing, which channel is which, expected
     object size.
   - **Visual check** *(non-blocking)* — after any step that changes how the data
     is interpreted. Layer to the viewer, screenshot, and report two or three
     numbers. **Never a screenshot alone**: without numbers an agent will call a
     failed result good. Sessions can be headless and volumes can be huge, so
     every visual check needs a numeric fallback and a stated slice or crop.
   - **Validate-and-gate** *(blocking)* — immediately before something expensive
     or hard to undo: scaling out over the catalog, a full-volume GPU op,
     declaring numbers final.

   Budget **at most three blocking checkpoints** in the body of a skill, and spend
   them where the next step costs far more than the question. Two things sit
   outside that budget: the final hand-off, which ends the workflow rather than
   interrupting it, and **anything destructive — which always asks first, no
   matter how many gates have been spent.** Restarting the kernel, interrupting a
   running job, overwriting a layer, and writing files all qualify.

5. **Dogfood it** *(visual check)*. Follow the drafted steps verbatim against
   real data, in the session you are already in — including the checkpoints.
   Anything you fix by improvising is a gap in the skill: fix the file, not the
   run. Then read the draft for hidden state: every variable and layer it uses
   must be created by its own steps, not left over from this conversation.

   A genuinely clean-room run needs `restart_kernel`, which **destroys the
   namespace, every layer, and any running job**. That is a *validate-and-gate*:
   ask the user first, explain what is lost, and offer to note the layers worth
   reloading. Never restart to test your own draft without consent.

6. **Check the draft yourself.** 

   - `id` is kebab-case and matches the intended filename stem.
   - `description` reads like the user's request; that string is what
     `find_skills` ranks on.
   - `tags` reuse tags you have seen on existing skills.
   - All six sections are present as `##` headings.
   - Every parameter has a derivation rule, not a magic constant.
   - The body is under roughly 200 lines — a proxy for the rule below.

   **Then ablate it.** A skill earns its length by fixing what a model gets wrong
   unaided, so measure that rather than guessing: give an subagent with fresh context
   the same task **without** the skill, and diff what it produces against what you
   wrote. Cut what it got right on its own; keep what it got wrong. Do not ask it
   which parts are obvious — models introspect their own competence badly.

   Run it a few times, and vary the scenario to put weight on particular advice —
   a deliberately oversized input is what shows whether a cost check is
   load-bearing. What it gets right only *sometimes* is where a one-line reminder
   beats a paragraph. A different model family is the better probe, since it does
   not share the blind spots of whoever drafted the skill.

7. **Deliver the draft** *(blocking)*. Show the complete file in the
   conversation. Then ask what the user wants done with it:

   - **Keep it for themselves** — save it as
     `~/.config/biopb/skills/<id>.md` (this is an explicit filesystem request, so
     it is allowed). It is picked up on the next `find_skills`, with no restart,
     and reported as `origin: local` so later sessions can tell it apart from a
     reviewed skill. Editing the file takes effect immediately too.
   - **Get it into the public catalog** — it goes through review by the biopb
     maintainers, who own the `biopb-site` repository. Offer to write a short
     summary alongside the file that they can send. Do not assume the user knows
     git or pull requests, and do not push anything yourself.

   Saving locally does not preclude publishing later; it is the same file.

## Guardrails

- **Do not restate the standing guardrails.** Repeating them in
   every skill guarantees they drift; state only where this skill *deviates*.
- **Do not reference dataset used during your work.** specific source_id, array_id and
   pathname prevent reuse.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| A saved skill never appears in `find_skills` | Saved outside `~/.config/biopb/skills`, or the filename starts with `_` (private), or skills are switched off | Check the path and `services.skills_enabled` |
| Section written as bold text or a deeper heading | Required sections are `##` headings | Use `## ` with the exact section names |
| The body has grown past ~200 lines | An algorithm is living in the skill | Move it to a plugin or a package pointer (step 2) |
| The body is long but every line reads as necessary | Never ablated — obviousness is invisible from the inside | Diff it against a cold small-model run (step 6) |

## Next steps

- A skill saved to `~/.config/biopb/skills` is live on the next `find_skills`,
  and stays personal and unreviewed (`origin: local`). Tell the user where it
  went, so they can edit or delete it without asking.
- A skill accepted by the maintainers is live within one deploy of the site —
  CI validates the frontmatter and the required sections at that point.
- Editing a published skill: change the body, bump `version`, leave `id` alone.
- Link related skills with `[[skill-id]]` so multi-step work composes into a
  chain, and only link ids that exist.
