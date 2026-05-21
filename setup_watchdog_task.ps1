# Run this script as Administrator to register the EA watchdog as a startup task
# Right-click PowerShell → "Run as Administrator" → paste this path and run

$pythonPath  = "C:\Users\khali\AppData\Local\Programs\Python\Python314\python.exe"
$watchdogPath = "C:\Users\khali\Documents\EA_Agent\watchdog.py"
$workDir     = "C:\Users\khali\Documents\EA_Agent"
$taskName    = "EA_Agent_Watchdog"

$action   = New-ScheduledTaskAction -Execute $pythonPath -Argument $watchdogPath -WorkingDirectory $workDir
$trigger  = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -StartWhenAvailable $true

$principal = New-ScheduledTaskPrincipal `
    -UserId "SYSTEM" `
    -LogonType ServiceAccount `
    -RunLevel Highest

Register-ScheduledTask `
    -TaskName    $taskName `
    -Action      $action `
    -Trigger     $trigger `
    -Settings    $settings `
    -Principal   $principal `
    -Description "Starts EA Bridge watchdog on system startup — restarts bridge if it crashes" `
    -Force

Write-Host ""
Write-Host "✅ Task '$taskName' registered successfully!" -ForegroundColor Green
Write-Host "   The watchdog will start automatically on next Windows boot."
Write-Host ""
Write-Host "To start it NOW without rebooting:"
Write-Host "   Start-ScheduledTask -TaskName '$taskName'"
