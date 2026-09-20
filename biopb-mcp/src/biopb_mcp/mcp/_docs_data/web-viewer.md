---
kind: reference
description: Show data in the browser when there is no napari window — the URL format, and who opens it.
---

# The web viewer

A browser page the control serves, reading the same tensor server the kernel
does. It is the display surface that does not depend on this session having a
window: it works on a headless box, over SSH, and on a machine the user is not
sitting at.

**The link is the whole interface.** Every viewing decision is in the URL — which
tensor, which plane, the contrast, the camera, which overlays — so one is a
complete instruction, and there is no session state to set up first.

Who opens it is a separate question with two answers, and a step usually wants
both:

- **The user**, by clicking it. This is what a visual check is *for*; opening
  the page yourself shows them nothing.
- **You**, if your host gives you browser automation. biopb ships no tool that
  drives the page, but most agents have one, and on a session with no napari
  window this is how you look at a result rather than only compute it.

## Building the link

```python
from biopb._endpoints import control_base_url
url = f"{control_base_url()}/viewer?id={array_id}"
```

`control_base_url()` is the loopback origin, `http://127.0.0.1:8813` unless the
control was moved. Take it from there rather than writing it out: the port is
configurable and `server_status` reports the resolved value under
`## Web viewer`.

A control published below the root (`--url-prefix`, an Open OnDemand job) still
answers at its own origin, so the link above works. What changes is the URL
*the user's browser* reaches it by — theirs carries the proxy's prefix before
`/viewer`. Ask them for it rather than guessing; nothing in this session can
discover it.

## Parameters

`id` is the tensor's whole address — `source_id`, or `source_id/field`, the same
`array_id` the catalog gives you. Everything else is optional and falls back to
the viewer's own default, so send the shortest link that says what you mean.

| | |
|---|---|
| `id` | the tensor to open |
| `t`, `z`, `c` | index along the named axes |
| `a<N>` | index along an unnamed axis, keyed by position (`a0`, `a3`, …) |
| `p` | auto-contrast percentile width, 0–4 (0 is min/max) |
| `cl=lo,hi` | a fixed contrast window instead, in the tensor's own grey levels |
| `g` | gamma |
| `v=1` / `v=0` | render as a volume, or as a plane |
| `vm` | volume mode: `mip`, `additive`, `minip` |
| `tg=x,y` or `tg=x,y,z` | camera target — two components is a plane, three a volume |
| `zm` | zoom, as log2 pixels per world unit |
| `rx`, `ro` | 3-D pitch (±90) and orbit (degrees) |
| `lb` | a label set drawn over the image, as *its* `array_id` |
| `lo` | the label overlay's alpha, 0–1 |
| `rs` | an annotation set to show; repeat it per set (`rs=default&rs=@ome`). Absent means the tensor's default, `rs=` alone means none |

Omitted means "the viewer's default", never "zero", so a link carries the intent
rather than a dump of the whole view. Parameters it does not recognise are left
alone, so you can hand a link back and forth without losing anything.

A camera is all-or-nothing on `tg` **and** `zm`: a target without a zoom frames
the volume somewhere nobody chose, so it is ignored. Send neither and the view
opens fitted, which is usually what you want.

## It shows the catalog, and only the catalog

This is the difference that changes how you work. A napari layer is this
session's own memory; the web viewer reads the tensor server, so **an array you
computed is invisible to it until you upload one**. See [[client]]:

```python
desc = client.create_tensor("cache:my_result", arr)
client.upload_array(desc, arr)
url = f"{control_base_url()}/viewer?id={desc.array_id}"
```

`cache:` is the right destination for something the user is only going to look
at. A segmentation is better uploaded as a *label set* of the image it came
from, which the viewer can then draw over the original with `lb=` — one link
instead of two, and the two stay registered.

Uploading costs a round trip over the data the user is about to look at, so for
a quick intermediate on a session that *does* have a napari window, the window
is cheaper. Where there is no window, this is the only route.

## Access

On a loopback control there is no token and the link works as written. Where the
control requires one (`--remote`), the user unlocks the page themselves; append
`&token=…` only if they gave you a token for this purpose. Do not go looking for
one.

## Opening it yourself

Where you have browser automation, this page is the replacement for
`take_screenshot` on a session with no napari window — `take_screenshot`
captures the napari canvas and cannot see a browser.

Three things to get right:

- **Give it time.** The page fetches tiles after it loads, so a capture taken
  the moment navigation finishes is of an empty canvas. Wait for the image, and
  re-capture rather than reporting the blank one.
- **Reachability is yours, not the user's.** The link is a loopback URL. Your
  browser tool can only open it if it runs on this machine; from elsewhere it
  will not resolve, which is a fact about where you run and not about the data.
- **Looking is not showing.** Your capture is yours. The user still needs the
  link, and a visual check is not satisfied by a screenshot they never saw.

## What it does not do

- **No 2-D annotation drawing from here.** ROIs are shown (`rs`), and are
  created through `client`, not by this page on your behalf.
- **Nothing in-memory.** No array, mask or overlay that is not a tensor on the
  server can be displayed.

## Related

- [[viewer]] — the napari window: the other display surface, and the one that
  can show an array without an upload.
- [[client]] — uploading a result so this page can read it.
- [[data]] — what an `array_id` addresses, and the pyramid behind it.
