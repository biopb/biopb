---
description: What a procedure doc must contain — the checkpoint types, and how to write a parameter.
---

# Writing a procedure doc

`write_doc`'s docstring holds the rules that decide *whether* to write. This is
what a procedure doc contains once you have.

## Sections

Four, as `##` headings: **When to use**, **When NOT to use**, **Parameters**,
**Steps**. *Failure modes* and *Next steps* are optional.

*When NOT to use* is the one most often skipped and the one an agent cannot
infer — do not apply background subtraction to a ratiometric image, do not
deconvolve before quantifying.

A *Failure modes* table is a record, not a forecast: add a symptom → cause → fix
row when a failure has actually been observed, and give the fix that worked. An
invented row is indistinguishable from an observed one once it is in the table,
and an agent matching a symptom to an invented cause debugs confidently in the
wrong direction. No section at all where nothing has failed yet.

## Parameters are derivation rules

Give each parameter as name, unit, and *how to derive the value from the data*.
`radius = 1.5x the largest object diameter` is usable; `radius = 50` is not — it
is this dataset's answer wearing a parameter's clothes.

## Checkpoints, by name

"Check with the user when appropriate" produces either constant interruption or
none at all. Name the type in the step:

- **confirm-input** *(blocking, before compute)* — only for facts the data
  cannot give you: voxel spacing, which channel is which, expected object size.
- **visual check** *(non-blocking)* — after any step that changes how the data
  is interpreted. Put the result where the user can see it and report two or
  three numbers with it. **Never a picture alone**: without numbers an agent
  will call a failed result good, and volumes can be too large to show usefully,
  so every visual check needs a numeric fallback and a stated slice or crop.

  *Where* they see it is the session's, not the doc's: the napari window when
  there is one ([[napari-viewer]]), otherwise a [[web-viewer]] link. Write the step as
  "show X" and name the crop and the numbers; do not write it as "add a layer",
  which is one of the two routes.

  Looking at it *yourself* is a third thing and not a substitute:
  `take_screenshot` on a napari window, or your host's browser automation on a
  [[web-viewer]] link. It is what stops you reporting a result you never saw —
  but the user has to see it too, so a doc says to show it either way.
- **validate-and-gate** *(blocking)* — immediately before something expensive or
  hard to undo: scaling out over the catalog, a full-volume GPU op, declaring
  numbers final.

At most **three blocking checkpoints** in a doc, spent where the next step costs
far more than the question. Outside that budget: the final hand-off, which ends
the workflow rather than interrupting it, and anything **destructive**, which
always asks first however many gates have been spent — restarting the kernel,
interrupting a running job, overwriting a layer, writing files.

## What belongs somewhere else

- **API mechanics.** Getting an array out of a layer or off the tensor server is
  [[tensor-server-client]]'s and [[napari-viewer]]'s job — pyramids, laziness, a
  per-axis `scale`, what an upload drops.
  A snippet here is a second copy that changes with the loader, and a wrong one
  runs and quietly reports the wrong numbers. Name what your steps need from the
  data ("both label arrays at the same level"), and link the reference.
- **Install mechanics.** Which command, and what a managed environment does to
  an added package, are in [[kernel]]. Write the half that is yours: which
  packages, and what the degraded path costs the result.
- **The standing guardrails.** Repeating them in every doc guarantees they
  drift. State only where this doc deviates.
- **This dataset.** A source_id, an array_id or a pathname makes the doc
  unusable by the next session. That run belongs in a notebook.

## How the code ships

| Amount | Ships as | The doc carries |
|---|---|---|
| ≲ 30 lines | Inline code fences | The code |
| 30–150 lines | A kernel plugin (`server_status` reports the dir under `## Kernel plugins`) | The call signature, qualified by the plugin's module name, and what the parameters mean |
| A published algorithm | A `pip install` pointer | Which package, and the degraded path when it is absent |

Step 1 of any doc with a requirement is the requirement check, before the
confirm-input step: there is no point asking which layer is truth if the scorer
was never going to be there. Keep the check to one sentence and name your own
fallback — the agent invents a worse one otherwise.

## Links

A doc links another as `[[id]]`, resolved with `read_doc(id)`. Link what this
doc depends on or hands off to, not everything it mentions. Index hooks carry no
links.

## Promotion

A doc you write is local and unreviewed. Getting it into the shipped set is a
pull request moving the identical file into `mcp/_docs_data/` in the `biopb`
repository, reviewed by the maintainers — offer to write the summary that goes
with it, and do not push anything yourself. The local copy keeps working until
then, and shadows the shipped one afterwards.
