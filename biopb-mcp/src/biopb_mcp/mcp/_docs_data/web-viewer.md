---
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
- **You**, through `show_view`, which moves the user's open tab to a view (and can
  return the image to you); or through your host's browser automation, if it has any.

## Building the link

```python
from biopb._control import user_base_url
url = f"{user_base_url()}/viewer?id={array_id}"
```

Behind a proxy (`--url-prefix`, an Open OnDemand job) the address the user's
browser reaches is not the one this machine connects to, so always take links for
the user from `user_base_url()`. `base_url()` is for your own requests, never a
link: behind a proxy it is loopback. `server_status` prints both under
`## Web viewer` when they differ.

`user_base_url()` can be a bare path (`/node/<host>/<port>`) when the proxy's public
origin was not configured: say so, or give the origin if you know it. It never
carries the access token.

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
| `tg=x,y` or `tg=x,y,z` | camera target — two components is a plane, three a volume; in pixels or voxels, not physical units ([camera](#the-camera)) |
| `zm` | zoom, as log2 pixels per world unit |
| `rx`, `ro` | 3-D pitch (±90) and orbit (degrees); part of the camera, so they need `tg` and `zm` ([camera](#the-camera)) |
| `lb` | a label set drawn over the image, as *its* `array_id` |
| `lo` | the label overlay's alpha, 0–1 |
| `rs` | an annotation set to show; repeat it per set (`rs=default&rs=@ome`). Absent means the tensor's default, `rs=` alone means none |

Omitted means "the viewer's default", never "zero", so a link carries the intent
rather than a dump of the whole view. Parameters it does not recognise are left
alone, so you can hand a link back and forth without losing anything.

Omitting the camera (`tg`, `zm`, `rx`, `ro`) opens the view fitted, which is usually
what you want; see [The camera](#the-camera) before setting one.

## Showing something you made

This is the difference that changes how you work. A napari layer is this
session's own memory; the web viewer reads the tensor server, so anything you
want on screen has to be **on the server** first. All three kinds can be, and
each has a parameter that draws it:

| what you have | show it with |
|---|---|
| an image, or any array uploaded as a tensor | `id=<array_id>` |
| a segmentation uploaded as a label set | `id=<image>&lb=<set array_id>` |
| points, boxes, polygons written with `put_rois` | `rs=<set_name>` |

```python
desc = client.setup_array_upload(f"zarr://{image_id}/@labels/nuclei", labels)
client.upload_array(desc, labels)
url = f"{user_base_url()}/viewer?id={image_id}&lb={desc.array_id}&lo=0.5"
```

[[upload]] is how each one gets there, and which to pick — a segmentation
belongs in a label set rather than its own tensor, so that one link shows both
and they stay registered.

Uploading costs a round trip over the data the user is about to look at, so on a
session that *does* have a napari window that window is cheaper for a quick
intermediate. Where there is no window, this is the route.

## Access

On a loopback control there is no token and the link works as written. Where the
control requires one (`--remote`), the user unlocks the page themselves; append
`&token=…` only if they gave you a token for this purpose. Do not go looking for
one.

## Presenting a result

`show_view(view)` is how a session with no napari window presents a result. It
moves the user's open viewer tab to `view` — the query string from
[Parameters](#parameters), `id` required — and returns at once. Like napari, it
changes the viewer: the view stays, so the user sees what you set. Overlays
(`lb`, `rs`) must already be on the server (above).

`show_view(view, image=True)` also returns a PNG of what the page drew, to check a
result before you report it; it waits for tiles and overlays, so it is slower.

| outcome | what you get | what to do |
|---|---|---|
| drawn | "Shown", or the PNG with `image=True` | go on; a note on the image means tiles or an overlay were still loading, so ask again rather than reporting it |
| not visible | the viewer moved, but its window is hidden (minimised, covered, another tab in front): no image | ask the user to bring it forward; the view is waiting |
| no tab | "Not shown: no viewer page is connected" | give the user the link (`user_base_url()`) and ask them to open it |

Every open viewer tab moves, since the control cannot tell which machine the user
is at; the answer is the first one on screen. Your own browser tool is the
alternative where the host gives you one and the link is reachable from where it
runs (the link is loopback); wait for the tiles before capturing, or you get an
empty canvas.

Looking is not showing: the user still needs the link for any visual check they
are meant to make.

## What it will not do

The page **displays; it does not author**. Nothing on it draws an ROI or edits a
label on your behalf — annotations are written with `client.put_rois` and read
back with `rs`. And it shows nothing that is not on the server, which is a step
(above) rather than a wall.

## The camera

A camera is all-or-nothing on `tg` **and** `zm`: a target without a zoom frames
the volume somewhere nobody chose, so it is ignored. Send neither and the view
opens fitted, which is usually what you want. **`rx` and `ro` belong to that
camera**: without `tg` and `zm` they are dropped, and a volume asked for at
`rx=30&ro=45` opens face-on.

**The target is in the viewer's own units, not µm.** On a plane it is image
pixels at full resolution, so `tg=128,128` is the middle of a 256 × 256 image.
On a volume it is the rendered volume's voxels with each axis stretched by its
physical size relative to the finest one: a 256 × 256 × 60 stack with 0.26 µm
pixels and 0.29 µm z-steps has its centre at about `tg=128,128,33.5`. Rather than
work that out, open the view, orbit it by hand, and read `tg`/`zm`/`rx`/`ro` off
the address bar.

## Related

- [[napari-viewer]] — the napari window: the other display surface, and the one that
  can show an array without an upload.
- [[upload]] — uploading a result so this page can read it.
- [[tensor-server-client]] — what an `array_id` addresses, and the pyramid
  behind it.
