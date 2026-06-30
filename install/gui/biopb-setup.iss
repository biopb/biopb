; ===========================================================================
; biopb Windows GUI installer  --  Inno Setup script (PROTOTYPE)
; ===========================================================================
; GUI front-end over the SAME headless engine the console installer uses
; (install/biopb-engine.ps1). It does NOT reimplement install logic: the wizard
; collects choices on pages, runs the engine as a hidden child process in
; -Mode gui, and renders the engine's tagged ::biopb:: progress stream LIVE.
;
; How live progress works (the crux):
;   Inno's Exec() blocks and cannot stream a child's stdout. So the engine is
;   told -LogFile <path>: it appends each tagged record to that file (no BOM,
;   UTF-8). We launch the engine with ewNoWait (non-blocking) and poll the file
;   on a Sleep loop, appending each new line to a scrolling log memo -- the
;   PRIMARY progress feedback -- under an indeterminate (marquee) bar plus a
;   one-line status. (We dropped the determinate n/total gauge: its updates were
;   finicky and could skip a step. The streaming console log -- every step plus
;   the raw uv/pip detail -- gives a truer sense of progress and surfaces the
;   diagnostics in view.) The engine's terminal ::biopb::DONE|<code> record ends
;   the loop. A full transcript is also written to <LogFile>.full.log.
;
; Compiles with Inno Setup 6.x (iscc). See docs/windows-installer.md.
;
; Design decisions (per planning):
;   * Online bootstrapper  -> ship the tiny engine .ps1; it downloads wheels.
;   * Per-user, no admin    -> PrivilegesRequired=lowest, install under %LOCALAPPDATA%.
; ===========================================================================

#define AppName        "biopb"
#define AppPublisher   "biopb"
#define AppURL         "https://biopb.org"
; AppVersion is injected by CI from the release-v* tag: iscc /DAppVersion=X.Y.Z
#ifndef AppVersion
  #define AppVersion   "0.0.0-dev"
#endif

[Setup]
AppId={{B10PB000-0000-0000-0000-000000000001}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
; Per-user, no admin: no UAC prompt, works on locked-down lab machines.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; Native x64 only. x64os refuses to run on ARM64 (and 32-bit x86), so we never
; reach a doomed install: key deps (pyarrow, the napari Qt stack) lack win-arm64
; wheels. InstallIn64BitMode makes the (always-32-bit) setup.exe run in 64-bit
; mode on x64, so Exec("powershell.exe") launches the 64-bit PowerShell rather
; than the WOW64 32-bit one (which reports PROCESSOR_ARCHITECTURE=x86). Needs
; Inno Setup 6.3+. (Requires Inno 6.3+ for the x64os identifier.)
ArchitecturesAllowed=x64os
ArchitecturesInstallIn64BitMode=x64os
DefaultDirName={localappdata}\biopb
DisableProgramGroupPage=yes
OutputBaseFilename=biopb-setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; CI signs the output (see docs/windows-installer.md). Once a signer is
; registered with iscc, uncomment to sign installer + uninstaller:
; SignTool=biopbsign

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

; No [Components]/[Types]: component selection was removed (biopb/biopb#237) --
; the web interface (now carrying the server admin page) is always installed and
; Bio-Formats is opt-in via $env:BIOPB_INSTALL_BIOFORMATS. The one remaining
; choice (remote-plugin consent, default on) is presented on a custom page (see
; [Code] InitializeWizard) with its description on the line beneath the checkbox,
; matching install.sh / the console front-end.

[Files]
; Stage the headless engine. The online model ships just this; everything else
; (uv, Python, wheels, napari) is fetched by the engine at run time.
Source: "..\biopb-engine.ps1"; DestDir: "{app}"; Flags: ignoreversion

[Code]
var
  OptionsPage:  TWizardPage;
  CbRemote:     TNewCheckBox;
  DataDirPage:  TInputDirWizardPage;
  ProgressPage: TOutputMarqueeProgressWizardPage;
  LogMemo:      TNewMemo;

  { Cloud / synced-folder support. When one or more cloud roots (OneDrive,
    iCloud) are found we add a checkbox to the data-dir page; ticking it points
    the picker at a Microscopy folder under a cloud root and passes -Cloud so the
    engine writes `cloud = true`. The engine ALSO auto-detects cloud-ness from the
    path, so the flag is belt-and-suspenders -- browsing to a OneDrive folder
    works regardless. With more than one signed-in OneDrive a dropdown
    (CbCloudCombo) lets the user pick which one (#188). }
  CbCloud:      TNewCheckBox;
  CbCloudDesc:  TNewStaticText;
  CbCloudCombo: TNewComboBox;
  CloudRoots:   TArrayOfString;
  LocalDataDir: String;

  { Existing-config detection (mirrors the console "keep current config" path) }
  ConfigPath:   String;
  ConfigExists: Boolean;
  KeepConfig:   Boolean;

  { Parser/loop state }
  CurStepText:  String;
  EngineDone:   Boolean;
  EngineExit:   Integer;
  LastError:    String;

  { Finish-page state, set from RESULT records }
  ResWebapp:    Boolean;
  ResMcpManual: Boolean;
  ResConfig:    String;

{ Win32 millisecond tick counter -- not a built-in Inno identifier. }
function GetTickCount: DWord;
  external 'GetTickCount@kernel32.dll stdcall';

{ Win32 SendMessage + scroll constants: TNewMemo has no built-in "scroll to
  bottom", so we post a WM_VSCROLL/SB_BOTTOM after each line for console-style
  auto-follow (the newest line stays visible). }
const
  WM_VSCROLL = $0115;
  SB_BOTTOM  = 7;
function SendMessage(hWnd: HWND; Msg: UINT; wParam: Longint; lParam: Longint): Longint;
  external 'SendMessageW@user32.dll stdcall';

{ Add one option to the custom page: a checkbox at the current Y, then its
  description wrapped on the line(s) directly underneath (indented under the
  checkbox text). Advances Top past both. Returns the checkbox so the caller can
  read .Checked later. }
function AddOption(var Top: Integer; const Title, Desc: String; Checked: Boolean; DescHeight: Integer): TNewCheckBox;
var
  cb:  TNewCheckBox;
  lbl: TNewStaticText;
begin
  cb := TNewCheckBox.Create(OptionsPage);
  cb.Parent  := OptionsPage.Surface;
  cb.Left    := 0;
  cb.Top     := Top;
  cb.Width   := OptionsPage.SurfaceWidth;
  cb.Height  := ScaleY(17);
  cb.Caption := Title;
  cb.Checked := Checked;
  Top := Top + cb.Height + ScaleY(2);

  lbl := TNewStaticText.Create(OptionsPage);
  lbl.Parent   := OptionsPage.Surface;
  lbl.Left     := ScaleX(18);   { indent under the checkbox text }
  lbl.Top      := Top;
  lbl.Width    := OptionsPage.SurfaceWidth - ScaleX(18);
  lbl.AutoSize := False;
  lbl.WordWrap := True;
  lbl.Height   := DescHeight;
  lbl.Caption  := Desc;
  Top := Top + DescHeight + ScaleY(12);

  Result := cb;
end;

{ Append P to Roots if it exists on disk and is not already present
  (case-insensitive). Trailing backslash is stripped so comparisons are stable. }
procedure AddCloudRoot(const P: String);
var
  q: String;
  i, n: Integer;
begin
  if P = '' then Exit;
  q := RemoveBackslashUnlessRoot(P);
  if not DirExists(q) then Exit;
  for i := 0 to GetArrayLength(CloudRoots) - 1 do
    if CompareText(CloudRoots[i], q) = 0 then Exit;   { dedupe }
  n := GetArrayLength(CloudRoots);
  SetArrayLength(CloudRoots, n + 1);
  CloudRoots[n] := q;
end;

{ Fill CloudRoots with every existing cloud / synced-folder root. Mirrors the
  engine's Get-CloudRoots probe order: OneDrive env vars, then the OneDrive
  registry accounts, then iCloud. The env vars only ever name ONE business
  account (the most recently active one), so the registry pass --
  HKCU\Software\Microsoft\OneDrive\Accounts\*\UserFolder, which lists every
  signed-in account (personal + each business) -- is the multi-OneDrive fix
  (#188). The engine additionally reads Dropbox's JSON sidecar, but OneDrive
  covers the lab norm and the engine still auto-detects a hand-picked Dropbox
  path by prefix. }
procedure DiscoverCloudRoots;
var
  accounts: TArrayOfString;
  i: Integer;
  uf, prof: String;
begin
  SetArrayLength(CloudRoots, 0);
  AddCloudRoot(GetEnv('OneDrive'));
  AddCloudRoot(GetEnv('OneDriveConsumer'));
  AddCloudRoot(GetEnv('OneDriveCommercial'));
  if RegGetSubkeyNames(HKEY_CURRENT_USER, 'Software\Microsoft\OneDrive\Accounts', accounts) then
    for i := 0 to GetArrayLength(accounts) - 1 do
      if RegQueryStringValue(HKEY_CURRENT_USER,
           'Software\Microsoft\OneDrive\Accounts\' + accounts[i], 'UserFolder', uf) then
        AddCloudRoot(uf);
  prof := GetEnv('USERPROFILE');
  if prof <> '' then AddCloudRoot(AddBackslash(prof) + 'iCloudDrive');
end;

{ The Microscopy folder under the Idx-th detected cloud root. }
function CloudDirFor(Idx: Integer): String;
begin
  Result := AddBackslash(CloudRoots[Idx]) + 'Microscopy';
end;

{ Index of the cloud root whose Microscopy folder equals P, or -1. Used to tell a
  still-default cloud suggestion apart from a hand-typed/browsed path. }
function IndexOfCloudDir(const P: String): Integer;
var
  i: Integer;
begin
  Result := -1;
  for i := 0 to GetArrayLength(CloudRoots) - 1 do
    if CompareText(P, CloudDirFor(i)) = 0 then begin Result := i; Exit; end;
end;

{ Currently selected cloud root (the dropdown choice, or the only root). }
function CurrentCloudIdx: Integer;
begin
  if CbCloudCombo <> nil then Result := CbCloudCombo.ItemIndex else Result := 0;
  if Result < 0 then Result := 0;
end;

{ Re-point the suggestion at the newly selected cloud root -- but only when the
  field still holds one of our cloud defaults, so a hand-typed path is kept. }
procedure CloudComboChange(Sender: TObject);
begin
  if CbCloud.Checked and (IndexOfCloudDir(DataDirPage.Values[0]) >= 0) then
    DataDirPage.Values[0] := CloudDirFor(CurrentCloudIdx);
end;

{ Toggle the suggested folder between the local default and a Microscopy folder
  under the chosen cloud root. Only swaps when the box still holds a KNOWN
  default (local on the way in, any cloud default on the way out), so a
  hand-typed/browsed path is never clobbered. }
procedure CloudCheckboxClick(Sender: TObject);
begin
  if CbCloudCombo <> nil then CbCloudCombo.Enabled := CbCloud.Checked;
  if CbCloud.Checked then begin
    if CompareText(DataDirPage.Values[0], LocalDataDir) = 0 then
      DataDirPage.Values[0] := CloudDirFor(CurrentCloudIdx);
  end else begin
    if IndexOfCloudDir(DataDirPage.Values[0]) >= 0 then
      DataDirPage.Values[0] := LocalDataDir;
  end;
end;

procedure InitializeWizard;
var
  T: Integer;
  i: Integer;
begin
  { Component selection is no longer offered (biopb/biopb#237): biopb-mcp, the
    data server, and the web interface (image viewer + server admin page) are
    always installed; Bio-Formats is opt-in only via $env:BIOPB_INSTALL_BIOFORMATS.
    The page now carries a single privacy choice -- the remote-plugin consent --
    with its description on the line(s) directly beneath the checkbox.
    ASCII-only text keeps the .iss codepage-safe. }
  OptionsPage := CreateCustomPage(wpWelcome,
    'Remote algorithm plugins',
    'biopb-mcp, the data server, and the web interface are always installed.');
  T := ScaleY(4);
  CbRemote := AddOption(T, 'Remote algorithm plugins',
    'Use off-site servers (hosted at UConn Health) for tasks like cell segmentation. Those servers log your IP address; uncheck to keep them disabled.',
    True, ScaleY(50));

  { Data-directory page -> engine -DataDir, placed right after the options page. }
  { Detect a previous install the same way the engine/console do: the config is
    at a fixed home-relative path (covers both GUI and `irm|iex` console
    installs). If present, we offer to keep it (see NextButtonClick). }
  { Canonical config is biopb.json (biopb/biopb#34); a legacy biopb.toml from a
    pre-#34 install still counts. Prefer the JSON path for display when present. }
  ConfigPath := AddBackslash(GetEnv('USERPROFILE')) + '.config\biopb\biopb.json';
  if not FileExists(ConfigPath) then
    ConfigPath := AddBackslash(GetEnv('USERPROFILE')) + '.config\biopb\biopb.toml';
  ConfigExists := FileExists(ConfigPath);
  KeepConfig   := False;

  DataDirPage := CreateInputDirPage(OptionsPage.ID,
    'Microscopy data directory',
    'Where are the images biopb should serve?',
    'biopb will index this folder. You can change it later in biopb.json.',
    False, '');
  DataDirPage.Add('');
  { Default under the profile root, NOT the Documents folder: Documents is
    frequently OneDrive-redirected, and OneDrive "Files On-Demand" placeholders
    hang the server's directory scan. Matches the console installer's fallback. }
  LocalDataDir := AddBackslash(GetEnv('USERPROFILE')) + 'Microscopy';
  DataDirPage.Values[0] := LocalDataDir;

  { If a cloud root exists, offer it: a checkbox under the dir input that points
    the picker at a Microscopy folder there and flags the source `cloud = true`.
    Cloud sources are now safe to index -- the server admits Files-On-Demand
    placeholders as unresolved sources (resolved lazily on read) instead of
    hanging the scan, which is exactly why we once steered AWAY from OneDrive. }
  DiscoverCloudRoots;
  if GetArrayLength(CloudRoots) > 0 then begin
    CbCloud := TNewCheckBox.Create(DataDirPage);
    CbCloud.Parent  := DataDirPage.Surface;
    CbCloud.Left    := 0;
    CbCloud.Top     := DataDirPage.Edits[0].Top + DataDirPage.Edits[0].Height + ScaleY(16);
    CbCloud.Width   := DataDirPage.SurfaceWidth;
    CbCloud.Height  := ScaleY(17);
    CbCloud.Caption := 'My images are in a cloud folder (OneDrive / iCloud / Dropbox)';
    CbCloud.OnClick := @CloudCheckboxClick;

    CbCloudDesc := TNewStaticText.Create(DataDirPage);
    CbCloudDesc.Parent   := DataDirPage.Surface;
    CbCloudDesc.Left     := ScaleX(18);
    CbCloudDesc.Top      := CbCloud.Top + CbCloud.Height + ScaleY(2);
    CbCloudDesc.Width    := DataDirPage.SurfaceWidth - ScaleX(18);
    CbCloudDesc.AutoSize := False;
    CbCloudDesc.WordWrap := True;
    CbCloudDesc.Height   := ScaleY(34);
    CbCloudDesc.Caption  := 'biopb indexes these without downloading every file -- images are ' +
                            'pulled on demand the first time you open them.';

    { Multiple signed-in OneDrives: a dropdown picks which one (env vars name
      only one business account -- #188). A single root needs no dropdown, so the
      common case keeps the bare-checkbox UX. Disabled until cloud is ticked. }
    if GetArrayLength(CloudRoots) > 1 then begin
      CbCloudCombo := TNewComboBox.Create(DataDirPage);
      CbCloudCombo.Parent  := DataDirPage.Surface;
      CbCloudCombo.Style    := csDropDownList;
      CbCloudCombo.Left     := ScaleX(18);
      CbCloudCombo.Top      := CbCloudDesc.Top + CbCloudDesc.Height + ScaleY(6);
      CbCloudCombo.Width    := DataDirPage.SurfaceWidth - ScaleX(18);
      for i := 0 to GetArrayLength(CloudRoots) - 1 do
        CbCloudCombo.Items.Add(CloudRoots[i]);
      CbCloudCombo.ItemIndex := 0;
      CbCloudCombo.Enabled   := False;
      CbCloudCombo.OnChange  := @CloudComboChange;
    end;
  end;

  { Progress page. We deliberately DROP the step-counting determinate gauge -- its
    n/total updates proved finicky and could skip a step -- in favor of an
    indeterminate MARQUEE page that simply animates "working...", with the REAL
    feedback being the scrolling log memo below. The memo streams the engine's full
    console output live (every step plus the raw uv/pip detail), so the user gets a
    genuine sense of progress and the diagnostic detail is in plain view instead of
    buried in the log file. The marquee is pumped via .Animate in the poll loop. }
  ProgressPage := CreateOutputMarqueeProgressPage('Installing biopb',
    'Setting up the biopb stack on your computer. This can take several minutes.');

  LogMemo := TNewMemo.Create(WizardForm);
  LogMemo.Parent     := ProgressPage.Surface;
  LogMemo.Left       := 0;
  LogMemo.Top        := ScaleY(76);
  LogMemo.Width      := ProgressPage.SurfaceWidth;
  LogMemo.Height     := ProgressPage.SurfaceHeight - ScaleY(76);
  LogMemo.ScrollBars := ssVertical;
  LogMemo.ReadOnly   := True;
  LogMemo.WordWrap   := False;
end;

function BuildEngineArgs(const LogPath: String): String;
var
  Args: String;
begin
  { Translate wizard choices into engine parameters. -Mode gui makes the engine
    emit the tagged ::biopb:: stream; -LogFile is the file we poll. }
  Args := '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\biopb-engine.ps1') + '"';
  Args := Args + ' -Mode gui';
  Args := Args + ' -LogFile "' + LogPath + '"';
  { Keep -> leave the existing config untouched (engine honors -KeepConfig);
    otherwise (re)write biopb.json pointing at the chosen folder. }
  if KeepConfig then
    Args := Args + ' -KeepConfig'
  else begin
    Args := Args + ' -DataDir "' + DataDirPage.Values[0] + '"';
    { Cloud opt-in (only meaningful when (re)writing the config). The engine also
      auto-detects cloud-ness from the path, so this is an explicit override. }
    if (CbCloud <> nil) and CbCloud.Checked then
      Args := Args + ' -Cloud';
  end;
  { The web interface is always installed now (it carries the server admin page);
    Bio-Formats is no longer a GUI option (opt in via $env:BIOPB_INSTALL_BIOFORMATS
    before launching, or rerun the console installer). }
  Args := Args + ' -Webapp';
  { Default ON; unchecking it disables the off-site cellpose server (IP logging). }
  if not CbRemote.Checked then Args := Args + ' -NoRemotePlugins';
#ifdef DryRun
  { Built with `iscc /DDryRun`: the engine walks the steps but changes nothing,
    so the whole wizard can be exercised safely. Absent in a normal build. }
  Args := Args + ' -DryRun';
#endif
  Result := Args;
end;

procedure AddLog(const S: String);
begin
  LogMemo.Lines.Add(S);
  { Console-style auto-follow: keep the newest line visible without the user
    having to scroll. }
  SendMessage(LogMemo.Handle, WM_VSCROLL, SB_BOTTOM, 0);
  { Mirror the most recent line as the one-line status above the log. }
  ProgressPage.SetText(CurStepText, S);
end;

procedure HandleResult(const Key, Val: String);
begin
  if      Key = 'webapp'     then ResWebapp := True
  else if Key = 'mcp_manual' then ResMcpManual := True
  else if Key = 'config'     then ResConfig := Val;
end;

{ Parse one line of the structured log. Tagged records are ::biopb::TAG|fields;
  anything else is treated as raw sub-command output. }
procedure HandleLine(const Line: String);
var
  body, tag, rest, rem, nStr, totStr, msg: String;
  p: Integer;
begin
  if Copy(Line, 1, 9) = '::biopb::' then
  begin
    body := Copy(Line, 10, Length(Line));
    p := Pos('|', body);
    if p > 0 then
    begin
      tag  := Copy(body, 1, p - 1);
      rest := Copy(body, p + 1, Length(body));
    end
    else
    begin
      tag  := body;
      rest := '';
    end;

    if tag = 'STEP' then
    begin
      { rest = n|total|msg. The marquee page has no determinate position to set;
        we render the step as a header line + the one-line status. }
      p := Pos('|', rest); nStr := Copy(rest, 1, p - 1); rem := Copy(rest, p + 1, Length(rest));
      p := Pos('|', rem);  totStr := Copy(rem, 1, p - 1); msg := Copy(rem, p + 1, Length(rem));
      CurStepText := '[' + nStr + '/' + totStr + ']  ' + msg;
      ProgressPage.SetText(CurStepText, '');
      AddLog(CurStepText);
    end
    else if tag = 'DONE' then
    begin
      EngineExit := StrToIntDef(rest, 1);
      EngineDone := True;
    end
    else if tag = 'RESULT' then
    begin
      p := Pos('|', rest);
      if p > 0 then HandleResult(Copy(rest, 1, p - 1), Copy(rest, p + 1, Length(rest)))
      else HandleResult(rest, '');
    end
    else if tag = 'ERROR' then
    begin
      LastError := rest;
      AddLog('ERROR: ' + rest);
    end
    else if tag = 'CMD' then
      AddLog('    ' + rest)
    else
      { OK / INFO / DETAIL / WARN / NOTE }
      AddLog(rest);
  end
  else
  begin
    { Untagged line = raw sub-command output. In -LogFile mode the structured
      file holds tagged lines only, so this rarely fires; kept for robustness. }
    if Trim(Line) <> '' then AddLog(Line);
  end;
end;

procedure RunEngine;
var
  LogPath: String;
  ResultCode, Processed, Cnt: Integer;
  Lines: TArrayOfString;
  StartTick: DWord;
begin
  LogPath := ExpandConstant('{tmp}\biopb-progress.log');
  DeleteFile(LogPath);

  EngineDone := False;
  EngineExit := 1;
  Processed  := 0;
  CurStepText := 'Starting the biopb installer...';

  ProgressPage.SetText(CurStepText, '');
  ProgressPage.Show;
  try
    if not Exec('powershell.exe', BuildEngineArgs(LogPath), '', SW_HIDE, ewNoWait, ResultCode) then
      RaiseException('Could not launch the biopb install engine (powershell.exe not found?).');

    StartTick := GetTickCount;
    repeat
      Sleep(200);
      { Advance the indeterminate marquee so it visibly animates during this
        blocking poll loop (the bar is not driven by a message pump here). }
      ProgressPage.Animate;
      try
        if LoadStringsFromFile(LogPath, Lines) then
        begin
          Cnt := GetArrayLength(Lines);
          while Processed < Cnt do
          begin
            HandleLine(Lines[Processed]);
            Processed := Processed + 1;
          end;
        end;
      except
        { File momentarily locked by the writer -- skip this tick, retry. }
      end;
      { Absolute backstop (90 min): the engine always emits DONE on success and
        failure, so we only hit this if powershell crashes hard. Note: Cancel is
        not serviced during this loop (see docs TODO). }
      if (GetTickCount - StartTick) > 5400000 then
      begin
        LastError := 'Timed out waiting for the install engine.';
        EngineExit := 1;
        EngineDone := True;
      end;
    until EngineDone;
  finally
    ProgressPage.Hide;
  end;

  if EngineExit <> 0 then
    RaiseException('biopb installation failed: ' + LastError + #13#10 +
      'Full log: ' + LogPath + '.full.log');
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  { ssPostInstall: the staged biopb-engine.ps1 now exists under the app dir, so
    drive the real install here and gate completion on the engine's exit code. }
  if CurStep = ssPostInstall then
    RunEngine;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  { Leaving the options page with an existing config present: ask whether to keep
    it -- the GUI equivalent of the console/Linux "Keep my current config file
    (default)" choice. Yes -> keep untouched and skip the data-dir page; No ->
    pick a new data folder (the engine preserves your other settings and replaces
    only the data folder). }
  if (CurPageID = OptionsPage.ID) and ConfigExists then
    KeepConfig := (MsgBox(
      'An existing biopb configuration was found:' + #13#10 +
      ConfigPath + #13#10#13#10 +
      'Keep your current configuration (data folder and settings)?' + #13#10#13#10 +
      'Yes  -  keep it unchanged' + #13#10 +
      'No   -  choose a new microscopy data folder (your other settings are kept)',
      mbConfirmation, MB_YESNO) = IDYES);
end;

{ Skip the data-directory page when the user chose to keep their current config. }
function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := (PageID = DataDirPage.ID) and KeepConfig;
end;

procedure CurPageChanged(CurPageID: Integer);
var
  Msg: String;
begin
  if CurPageID = wpFinished then
  begin
    Msg := 'biopb is installed. Start your AI agent (Claude Code/Desktop, Cursor,' + #13#10 +
           'opencode) and a napari window opens with it.';
    if ResWebapp then
      Msg := Msg + #13#10#13#10 + 'Web interface: http://localhost:8814';
    if ResConfig <> '' then
      Msg := Msg + #13#10#13#10 + 'Config: ' + ResConfig;
    if ResMcpManual then
      Msg := Msg + #13#10#13#10 + 'NOTE: no MCP client was detected -- register biopb manually' + #13#10 +
             'using ' + ExpandConstant('{userappdata}') + '\..\.config\biopb\mcp.json';
    WizardForm.FinishedLabel.Caption := Msg;
  end;
end;

{ ---- Uninstall: drive the engine's -Uninstall mode ----
  The real install lives under %USERPROFILE% (uv tool env, .local\bin shims,
  .config, .local\share), NOT under the app dir, so removing the app dir alone
  would orphan it. usUninstall runs before files are deleted, so the staged
  engine is still present. We prompt before purging config/cached data; images
  are never touched. The .local\bin PATH entry is shared with uv and left alone. }
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  enginePath, purgeArg: String;
  ResultCode: Integer;
begin
  if CurUninstallStep <> usUninstall then Exit;

  purgeArg := '';
  if MsgBox('Also remove biopb configuration and cached data?' + #13#10 +
            '(config files, the web interface, caches and logs)' + #13#10#13#10 +
            'Your microscopy images are NOT affected.',
            mbConfirmation, MB_YESNO) = IDYES then
    purgeArg := ' -Purge';

  enginePath := ExpandConstant('{app}\biopb-engine.ps1');
  if FileExists(enginePath) then
    Exec('powershell.exe',
         '-NoProfile -ExecutionPolicy Bypass -File "' + enginePath + '" -Uninstall -Mode console' + purgeArg,
         '', SW_HIDE, ewWaitUntilTerminated, ResultCode)
  else
    { Fallback if the staged engine is missing (very old install): at least drop
      the uv tool env so the packages do not linger. }
    Exec('powershell.exe',
         '-NoProfile -ExecutionPolicy Bypass -Command "uv tool uninstall biopb"',
         '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;
