; Inno Setup script for ND Lore Forge. Built by packaging/windows/build.ps1 (iscc /DAppVersion=0.1.0 ndforge.iss).
; The in-app updater runs this installer again over the installed program: /SILENT /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS /NORESTART. The same AppId (below, never change it) makes
; that an upgrade in place: same folder, same per-user / all-users choice, shortcuts and uninstaller kept; the worker and tools folders are replaced, the user's projects and settings are not in {app}.
; Installs per user by default (no administrator needed); a user can choose "for all users" in the dialog.
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#define AppName "ND Lore Forge"

[Setup]
AppId={{6F1D8A52-3C0E-4B0B-9E5B-2E6F7D1A0C11}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=ND Lore Forge
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UsePreviousPrivileges=yes
CloseApplications=yes
OutputDir=..\..\dist
OutputBaseFilename=ND-Lore-Forge-Setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\nd-lore-forge.exe
SetupIconFile=..\..\src-tauri\icons\icon.ico

[Files]
Source: "..\..\target\release\nd-lore-forge.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\dist\stage\worker\*"; DestDir: "{app}\worker"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\..\dist\stage\tools\*"; DestDir: "{app}\tools"; Flags: ignoreversion
Source: "..\..\README.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\nd-lore-forge.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\nd-lore-forge.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Run]
Filename: "{app}\nd-lore-forge.exe"; Description: "Start {#AppName}"; Flags: nowait postinstall skipifsilent
; a silent upgrade started with /RESTARTAPPLICATIONS (the in-app updater) starts the program again; the program allows one instance, so a second start does no harm
Filename: "{app}\nd-lore-forge.exe"; Flags: nowait skipifnotsilent; Check: RestartRequested

[Code]
function RestartRequested(): Boolean;
var i: Integer;
begin
  Result := False;
  for i := 1 to ParamCount do
    if CompareText(ParamStr(i), '/RESTARTAPPLICATIONS') = 0 then Result := True;
end;

// The window is drawn by Microsoft Edge WebView2 (already on Windows 11 and on most up-to-date Windows 10). Tell the user if it is missing.
function WebView2Installed(): Boolean;
var v: String;
begin
  Result := RegQueryStringValue(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', v)
         or RegQueryStringValue(HKCU, 'Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', v);
  if Result then Result := (v <> '') and (v <> '0.0.0.0');
end;

function InitializeSetup(): Boolean;
begin
  Result := True;
  if not WebView2Installed() then
    MsgBox('ND Lore Forge needs the Microsoft Edge WebView2 runtime, which was not found on this computer.' + #13#10 + #13#10 +
           'After the installation, if the window does not open, install it from' + #13#10 +
           'https://developer.microsoft.com/microsoft-edge/webview2/ and start the program again.', mbInformation, MB_OK);
end;
