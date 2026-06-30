# Windows GUI installer

How the biopb Windows installer is structured so a GUI wizard and the existing
console one-liner share **one install brain**.

> Status: **implemented and shipping.** The engine (`install/biopb-engine.ps1`),
> console front-end (`install/install.ps1`), and Inno Setup GUI wizard
> (`install/gui/biopb-setup.iss`) are all on `dev` and drive one shared engine; the
> Windows job in `release.yaml` builds the installer. Code signing is the one
> deferred piece (no cert yet). (Was a prototype on `feat/windows-gui-installer`.)

## The core constraint

`install.ps1` is **not a file-copy install** — it is a multi-minute network
*orchestration*: install `uv`, provision Python ≥3.10, download the wheel triple
from the latest `release-v*`, pull `napari[all]` from PyPI, unpack the webapp,
write `biopb.json`, start the tensor server, and wire MCP clients. It is
idempotent and already handles the interactive choices.

That rules out a classic **MSI**, whose declarative file-transaction model fights
a long, network-dependent Python bootstrap (rollback semantics, no real progress,
fragile deferred custom actions). The GUI must **wrap the orchestration we already
have**, not replace it.

## Decisions (locked for v1)

| Decision | Choice | Consequence |
|---|---|---|
| Distribution | **Online bootstrapper** | Ship a tiny signed `.exe` that stages the engine `.ps1`; the engine downloads wheels at run time (same as today). No fat bundle to sign per-arch. |
| Scope | **Per-user, no admin** | `%USERPROFILE%` / `%LOCALAPPDATA%`, user PATH, no UAC. Works for grad students on locked-down machines. → **Inno Setup**, not WiX/MSI. |
| Architecture | **Native x64 only** | ARM64 is rejected (not silently attempted): key deps (pyarrow, the napari Qt stack) lack win-arm64 wheels, so a native ARM64 install would fail downstream. Inno `ArchitecturesAllowed=x64os` refuses ARM64/x86 at launch; the engine also gates on the true arch. |

**WOW64 note:** Inno's `setup.exe` is always 32-bit, so a launched `powershell.exe`
gets WOW64-redirected to the 32-bit PowerShell, where `PROCESSOR_ARCHITECTURE`
reads `x86` even on x64. Two defenses: `ArchitecturesInstallIn64BitMode=x64os`
makes Exec launch the **64-bit** PowerShell, and the engine reads
`PROCESSOR_ARCHITEW6432` (the true arch) before falling back to
`PROCESSOR_ARCHITECTURE`.

Per-machine / GPO mass-deployment for managed lab fleets (WiX/MSI) is explicitly
*out of scope for v1* — kept in reserve for that audience.

## Architecture: one engine, two front-ends

```
                 choices (components, data dir)
                          │
   ┌──────────────────────┴───────────────────────┐
   │                                               │
install.ps1 (console front-end)        biopb-setup.iss (Inno GUI front-end)
   │  banner, preflight, prompts            │  welcome, components page, dir page
   │  dot-sources engine, -Mode console     │  runs engine as child, -Mode gui
   └──────────────────────┬────────────────┘
                          │  calls Invoke-BiopbInstall / runs the .ps1
                          ▼
              install/biopb-engine.ps1  (headless engine)
        uv → Python → wheel triple → webapp → config → server → MCP
        every choice is a PARAMETER; no Read-Host / banner / exit
        emits progress through ONE reporter (console-colored OR tagged)
```

Both front-ends go through the same `Report-*` calls, so they can never drift on
*what* gets reported — only the rendering differs.

### Files

| File | Role |
|---|---|
| `install/biopb-engine.ps1` | Headless engine. Dual-use: **dot-sourced** (defines functions only — guarded by `$MyInvocation.InvocationName`) or **run directly** (executes in `-Mode gui`). |
| `install/install.ps1` | Console front-end. Banner, preflight, `Select-Components`, `Select-DataDir`, then drives the engine `-Mode console` in-process and renders the summary. |
| `install/gui/biopb-setup.iss` | Inno Setup wizard skeleton. Components page → `-Webapp`/`-Bioformats`; dir page → `-DataDir`; runs the engine and parses its tagged stream. |

## The progress protocol (the integration seam)

In `-Mode gui` the engine emits one tagged record per line. It writes to **stdout**
*and*, when given `-LogFile <path>`, appends each record to that file (no BOM,
UTF-8). The file is the transport the **Inno wizard polls** — `Exec()` can't
stream a child's stdout, but it can tail a file. A full transcript (incl. raw uv
output) is written alongside as `<LogFile>.full.log` for diagnosing failures.
Untagged lines are raw sub-command output the GUI shows as log detail.

```
::biopb::STEP|<n>|<total>|<message>     # drives the progress bar + step label
::biopb::OK|<message>                   # success line
::biopb::INFO|<message>                 # headline info
::biopb::DETAIL|<message>               # indented sub-line
::biopb::WARN|<message>
::biopb::NOTE|<message>
::biopb::CMD|<message>                  # a command the user can copy
::biopb::ERROR|<message>
::biopb::RESULT|<key>|<value>           # for the finish page (e.g. webapp|1)
::biopb::DONE|<exitcode>                # terminal record; nonzero => failure
```

In `-Mode console` the exact same `Report-*` calls render with the original
colored `Write-Host` styling, so existing `irm | iex` users see no regression.

## `irm | iex` is preserved

The console front-end stays a single self-contained entry point. When run from a
local checkout/unpacked installer it dot-sources the sibling `biopb-engine.ps1`;
when run via `irm | iex` (no file on disk, `$PSScriptRoot` empty) it downloads the
engine from `https://biopb.org/biopb-engine.ps1`. CI publishes the engine as a
release asset alongside `install.ps1`, and `biopb.org` serves it — mirroring how
`install.ps1`/`install.sh` are already hosted (see `release-model.md`).

> Alternative if the extra fetch is unwanted: a build step concatenates engine +
> front-end into one self-contained published `install.ps1`, with the engine also
> shipped standalone for Inno. The source stays decomposed either way.

## Inno wizard mapping

| Wizard page | Feeds | Replaces |
|---|---|---|
| Welcome / license | — | — |
| Options (custom page) | `-Webapp`, `-Bioformats`, `-NoRemotePlugins` | `Select-Components` + remote-plugins consent |
| Keep-config dialog | `-KeepConfig` (or proceed to data dir) | `Select-DataDir -KeepOption` |
| Data directory | `-DataDir` (`-Cloud` when the cloud box is ticked) | `Select-DataDir` |
| Progress | parses `STEP`/log records | the console `[n/7]` output |
| Finish | `RESULT` records | the console summary |

**Existing config / keep behavior.** On leaving the Options page the wizard
checks for `%USERPROFILE%\.config\biopb\biopb.json` (canonical, biopb/biopb#34),
falling back to a legacy `biopb.toml` (fixed paths, so it catches both prior GUI
*and* `irm|iex` console installs). If present, a Yes/No dialog offers to keep the
current configuration — the GUI equivalent of the console/Linux "Keep my current
config file (default)". **Yes** passes `-KeepConfig` (engine leaves the existing
config untouched) and skips the data-dir page; **No** shows the data-dir page and
the engine writes `biopb.json`, preserving the prior server/cache settings and
replacing only the `sources` list (a legacy `biopb.toml` is migrated to JSON and
backed up). We do not pre-read the existing data dir (a config may hold multiple
`sources` and the engine replaces them with the chosen folder, so "keep" means
*don't touch the file*).
The data-dir default is `%USERPROFILE%\Microscopy`, not Documents, matching the
console's OneDrive-avoiding choice.

**Cloud / synced folders.** When a cloud root is present the data-dir page adds a
*"My images are in a cloud folder"* checkbox: ticking it points the picker at a
`Microscopy` folder under the cloud root and passes `-Cloud`, which makes the
engine write `cloud = true` on the source. The GUI discovers roots the same way
the engine's `Get-CloudRoots` does — OneDrive env vars (`%OneDrive%`,
`%OneDriveConsumer/Commercial%`), **plus the OneDrive account registry**
(`HKCU\Software\Microsoft\OneDrive\Accounts\*\UserFolder`), then
`%USERPROFILE%\iCloudDrive`. The env vars only name one business account, so the
registry pass is what surfaces *every* signed-in OneDrive; when more than one is
found the checkbox is paired with a dropdown to pick which root (#188). The
console front-end gets
the same cloud folders for free — `Get-DataDirCandidates` (in the engine) now
offers them alongside the local/drive candidates. Either way the engine
*auto-detects* cloud-ness from the chosen path (`Test-IsCloudPath`: any dir at or
under a known cloud root), so browsing/typing a OneDrive path yields `cloud = true`
even without the checkbox; `-Cloud` is just an explicit override for roots the
probes miss (e.g. a Dropbox path the GUI didn't enumerate). A `cloud = true`
source admits OneDrive/Dropbox/iCloud "Files-On-Demand" placeholders as
*unresolved* sources (resolved lazily on first read) instead of letting the
directory scan hang on hydrate-on-read placeholders — the reason the installer
historically steered *away* from OneDrive. See the tensor server's cloud-storage
phase 2 (`SourceConfig.cloud`).

Inno gives the **uninstaller + Add/Remove Programs entry for free** — something
the CLI install can't offer today.

## Local build & test

`iscc` (Inno's compiler) installs per-user at
`%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe`.

**Build the installer:**

```powershell
$iscc = "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
& $iscc install\gui\biopb-setup.iss          # -> install\gui\Output\biopb-setup-0.0.0-dev.exe
```

**Test safely with `-DryRun`** — the highest-value local test. The engine has a
`-DryRun` switch that walks all 7 steps and emits the full progress stream
(console text or tagged `::biopb::` records) **without downloading or installing
anything, and without touching config/server state**. Use it to exercise the
console output and the whole GUI wizard on your own machine with zero impact.

```powershell
# Console front-end path (engine, -Mode console):
powershell -NoProfile -ExecutionPolicy Bypass -File install\biopb-engine.ps1 -Mode console -DryRun -Webapp

# GUI wizard: build a dry-run installer, then run it and walk the pages.
& $iscc /DDryRun install\gui\biopb-setup.iss
.\install\gui\Output\biopb-setup-0.0.0-dev.exe
```

A `/DDryRun` build passes `-DryRun` to the engine (via an `#ifdef` in
`biopb-setup.iss`); a normal build never does. Watch for: the data-dir page
default, the gauge advancing 0→7, the log memo scrolling, and the finish page
picking up the `RESULT` records.

**Full end-to-end test** (real install — downloads uv/Python/napari/wheels and
writes config): build without `/DDryRun` and run the `.exe`. The per-user/no-admin
design keeps it inside your profile, but do this in **Windows Sandbox or a VM**
for a clean, throwaway environment.

## Open items (plan now)

- **Code signing / SmartScreen — highest lead time, start immediately.** An
  unsigned installer hits a "Unknown publisher" SmartScreen wall (fatal trust hit
  for cautious lab users). The long pole is **legal identity validation**, not
  cert issuance. See [Code signing](#code-signing) below for the full process.
- **Uninstaller — implemented** via an engine `-Uninstall` mode (teardown runs
  through the same brain). The Inno uninstaller's `CurUninstallStepChanged`
  prompts "also remove config + cached data?" and runs the staged engine with
  `-Uninstall [-Purge]`. The engine stops the server, `uv tool uninstall biopb`,
  deregisters MCP clients (`Remove-McpClients`), and with `-Purge` deletes
  `.config\biopb`, `.config\biopb-mcp`, `.local\share\biopb`. Deliberately **not**
  removed: the `.local\bin` PATH entry (shared with uv) and the user's images.
  Crucially this targets the `%USERPROFILE%` install, not just the GUI's app dir.
- **Live progress plumbing in Inno — implemented** (`biopb-setup.iss`). The
  wizard launches the engine with `ewNoWait` + `-LogFile`, then polls the
  structured file on a `Sleep` loop at `ssPostInstall`, parsing `::biopb::`
  records into the gauge, status lines, and a scrolling log memo; the terminal
  `DONE|<code>` ends the loop. Remaining rough edges: (1) **Cancel is not
  serviced during the poll loop** (the loop blocks the UI thread) — needs a
  message-pump or worker-thread approach; (2) raw uv output isn't shown in the
  live memo (it's in `<LogFile>.full.log`); (3) memo auto-scroll-to-bottom not
  wired. A 90-min absolute timeout backstops a hard powershell crash.
- **CI — implemented (unsigned).** `release.yaml` has a `windows-installer` job
  (on `windows-latest`) that installs Inno Setup, runs
  `iscc /DAppVersion=<X.Y.Z>`, and uploads the `.exe` as an artifact; the
  `release` job downloads it and attaches `biopb-setup-<R>.exe` to the
  `release-v*` release next to the engine and `install.ps1`. **Signing is not yet
  wired** — no cert is provisioned, so the build is unsigned and SmartScreen warns
  "Unknown publisher". Once a cloud-HSM signer is registered with `iscc`, add the
  sign step (and uncomment `SignTool=biopbsign` in `biopb-setup.iss`).

## Phasing

- **Phase 0 (now, code-independent):** procure the signing cert (EV vs OV).
- **Phase 1 (done in this prototype):** engine ↔ console front-end split; console
  path unchanged for existing users.
- **Phase 2:** flesh out the Inno wizard + uninstaller against the engine.
- **Phase 3:** CI build + release-asset attachment — **done unsigned**; signing
  still pending the cert (Phase 0).

## Code signing

The single highest-lead-time item. Start the **identity** track now, in parallel
with the code work — the cert itself issues quickly once identity clears.

### Key fact: keys must live on hardware (both OV and EV)

Since June 2023 the CA/Browser Forum requires **all** code-signing private keys —
OV *and* EV — to be generated and stored on certified hardware (FIPS 140-2 Level 2
token or a cloud HSM). The old "download a `.pfx`, sign anywhere" flow is gone for
both tiers. So OV vs EV is now a **trust/price** decision, not a key-storage one.

### OV vs EV

| | OV | EV |
|---|---|---|
| SmartScreen | reputation builds over downloads — early users still warned | **instant trust**, no warning |
| Key storage | hardware token or cloud HSM | hardware token or cloud HSM (same) |
| Cost/yr | ~$200–400 | ~$300–600 |
| Lead time | days–2 weeks | 1–3 weeks |

For a low-volume scientific installer where first impressions matter, **EV** is
usually worth it — it skips the "accumulate downloads while everyone sees a
scary wall" reputation phase entirely.

### Identity path: university / institution (chosen, pending IT confirmation)

biopb will most likely sign under the **university/institution** (UCHC). This is
often the cleanest path for academic software — the institution may already hold
a code-signing cert, a D-U-N-S number, and an established validation history.

**Questions to confirm with institutional IT / research computing:**
- Does the institution already have an OV/EV code-signing certificate we can sign
  releases under, or a process to request one?
- Is the key on a **physical token** (blocks headless CI) or a **cloud HSM /
  signing service** (works in GitHub Actions)? Cloud HSM is required for our
  `release.yaml` to sign automatically.
- Who authorizes signing on the institution's behalf, and what's the turnaround?
- Any policy constraints on signing open-source / externally-distributed
  binaries under the institutional identity?

Fallbacks if the institutional route stalls: **register biopb as an org** (LLC/
nonprofit + free D-U-N-S from Dun & Bradstreet — itself a days–weeks long pole),
or an **individual developer cert** (Certum / SSL.com — cheapest, hardware-backed,
weaker branding, limited EV).

### Process once identity is settled

1. **Pick a CA** — DigiCert, Sectigo, SSL.com, Certum, GlobalSign. Or **Azure
   Trusted Signing** (~$10/mo, Microsoft-run, cloud-based, CI-friendly, confers
   SmartScreen trust) — check current org/individual eligibility; cheapest
   credible option if we qualify.
2. **Choose key storage = cloud HSM**, not a physical token — `release.yaml`
   signs headlessly, so a token someone must plug in is a non-starter. Options:
   DigiCert KeyLocker, SSL.com eSigner, Azure Trusted Signing.
3. **Validate identity** — org/business records, the D-U-N-S number, and a
   verification phone call to a publicly-listed number. This is the slow part.
4. **Receive credentials** — cloud-HSM API credentials + a cert reference; no
   `.pfx` touches disk.
5. **Sign + always timestamp** so signatures survive cert expiry:
   - token: `signtool sign /tr http://timestamp.digicert.com /td sha256 /fd sha256 biopb-setup.exe`
   - cloud HSM in CI: a wrapper like **AzureSignTool** / the CA's signer
     authenticates to the HSM and signs — no local key.
6. **Inno integration** — register a sign command in the Inno compiler config and
   reference it via the `SignTool=` directive in `biopb-setup.iss` (currently
   commented as `SignTool=biopbsign`). Inno then signs **both** the installer and
   the uninstaller as part of `iscc`.
7. **CI** — `release.yaml`: `iscc /DAppVersion=X.Y.Z` → sign via cloud HSM (creds
   from GitHub Actions secrets, never in the script) → attach
   `biopb-setup-<R>.exe` to the `release-v*` release.
