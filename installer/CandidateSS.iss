; CandidateSS installer for Windows (Inno Setup 6)

#define MyAppName "CandidateSS"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Candidate Screening System"
#define MyAppExeName "CandidateSS.exe"

#ifndef AppBuildMode
  #define AppBuildMode "onedir"
#endif

[Setup]
AppId={{4FA6768D-CC0A-4D17-A593-0C2D131E7F40}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=CandidateSS_Installer
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
#if AppBuildMode == "onedir"
Source: "..\dist\candidate_ss_gui.dist\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
#else
Source: "..\dist\CandidateSS.exe"; DestDir: "{app}"; Flags: ignoreversion
#endif

[Dirs]
Name: "{app}\incoming_media"
Name: "{app}\output"

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Удалить {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Создать ярлык на рабочем столе"; GroupDescription: "Дополнительные задачи:"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Запустить {#MyAppName}"; Flags: nowait postinstall skipifsilent
