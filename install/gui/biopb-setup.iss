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
;   on a Sleep loop, parsing new lines into the progress gauge, the status text,
;   and a scrolling log memo. The engine's terminal ::biopb::DONE|<code> record
;   ends the loop. A full transcript (incl. uv output) is written to
;   <LogFile>.full.log for diagnosing failures.
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

; No [Components]/[Types]: the options are presented on a custom page (see
; [Code] InitializeWizard) so each checkbox carries its OWN description on the
; line directly beneath it, rather than one shared block of text. Defaults
; (viewer on, Bio-Formats off, remote plugins on) match install.sh / the console.

[Files]
; Stage the headless engine. The online model ships just this; everything else
; (uv, Python, wheels, napari) is fetched by the engine at run time.
Source: "..\biopb-engine.ps1"; DestDir: "{app}"; Flags: ignoreversion

[Code]
var
  OptionsPage:  TWizardPage;
  CbViewer:     TNewCheckBox;
  CbBioformats: TNewCheckBox;
  CbRemote:     TNewCheckBox;
  DataDirPage:  TInputDirWizardPage;
  ProgressPage: TOutputProgressWizardPage;
  LogMemo:      TNewMemo;

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

procedure InitializeWizard;
var
  T: Integer;
begin
  { Options page: each checkbox carries its own description on the line(s)
    directly beneath it (TNewCheckBox + TNewStaticText pairs), instead of one
    shared block. biopb-mcp and the data server are always installed. Defaults
    match install.sh / the console (viewer on, Bio-Formats off, remote on).
    ASCII-only text keeps the .iss codepage-safe. }
  OptionsPage := CreateCustomPage(wpWelcome,
    'Installation options',
    'biopb-mcp and the data server are always installed. Choose the optional pieces below.');
  T := ScaleY(4);
  CbViewer := AddOption(T, 'Built-in data viewer',
    'Browse all your images in any web browser (Chrome, Safari, and others).',
    True, ScaleY(18));
  CbBioformats := AddOption(T, 'Bio-Formats support',
    'Adds many extra and proprietary image formats. Needs Java and some setup on first run, so it is off by default.',
    False, ScaleY(34));
  CbRemote := AddOption(T, 'Remote algorithm plugins',
    'Use off-site servers (hosted at UConn Health) for tasks like cell segmentation. Those servers log your IP address; uncheck to keep them disabled.',
    True, ScaleY(50));

  { Data-directory page -> engine -DataDir, placed right after the options page. }
  DataDirPage := CreateInputDirPage(OptionsPage.ID,
    'Microscopy data directory',
    'Where are the images biopb should serve?',
    'biopb will index this folder. You can change it later in biopb.toml.',
    False, '');
  DataDirPage.Add('');
  DataDirPage.Values[0] := ExpandConstant('{userdocs}\Microscopy');

  { Custom progress page: gauge + two status lines (native) plus a scrolling log
    memo we parent onto the page surface, below the gauge. }
  ProgressPage := CreateOutputProgressPage('Installing biopb',
    'Setting up the biopb stack on your computer.');

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
  Args := Args + ' -DataDir "' + DataDirPage.Values[0] + '"';
  if CbViewer.Checked     then Args := Args + ' -Webapp';
  if CbBioformats.Checked then Args := Args + ' -Bioformats';
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
  { Mirror the most recent line under the gauge so the user always sees activity
    even without scrolling the memo. }
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
      { rest = n|total|msg }
      p := Pos('|', rest); nStr := Copy(rest, 1, p - 1); rem := Copy(rest, p + 1, Length(rest));
      p := Pos('|', rem);  totStr := Copy(rem, 1, p - 1); msg := Copy(rem, p + 1, Length(rem));
      ProgressPage.SetProgress(StrToIntDef(nStr, 0), StrToIntDef(totStr, 7));
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
  ProgressPage.SetProgress(0, 7);
  ProgressPage.Show;
  try
    if not Exec('powershell.exe', BuildEngineArgs(LogPath), '', SW_HIDE, ewNoWait, ResultCode) then
      RaiseException('Could not launch the biopb install engine (powershell.exe not found?).');

    StartTick := GetTickCount;
    repeat
      Sleep(200);
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

{ Surface the engine's results on the final page. }
function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if CurPageID = wpFinished then
    { no-op hook; reserved for opening the data browser, etc. }
    ;
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
      Msg := Msg + #13#10#13#10 + 'Data browser: http://localhost:8815';
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
            '(config files, the data browser, caches and logs)' + #13#10#13#10 +
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
