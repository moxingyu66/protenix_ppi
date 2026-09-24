param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\artifacts\env")
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputFile = Join-Path $OutputDirectory "environment_windows_$timestamp.txt"
$report = [System.Collections.Generic.List[string]]::new()

function Add-Section {
    param(
        [string]$Title,
        [scriptblock]$Command
    )
    $report.Add("")
    $report.Add("===== $Title =====")
    try {
        $value = & $Command 2>&1 | Out-String
        $report.Add($value.TrimEnd())
    }
    catch {
        $report.Add("UNAVAILABLE: $($_.Exception.Message)")
    }
}

$report.Add("Protenix-PPI G0 environment report")
$report.Add("Collected: $(Get-Date -Format o)")
$report.Add("Review and redact hostnames, usernames, IP addresses, private paths, and credentials before sharing.")

Add-Section "Windows version" {
    Get-CimInstance Win32_OperatingSystem |
        Select-Object Caption, Version, OSArchitecture, TotalVisibleMemorySize, FreePhysicalMemory |
        Format-List
}

Add-Section "CPU" {
    Get-CimInstance Win32_Processor |
        Select-Object Name, NumberOfCores, NumberOfLogicalProcessors |
        Format-List
}

Add-Section "NVIDIA summary" {
    & nvidia-smi
}

Add-Section "NVIDIA inventory" {
    & nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
}

Add-Section "GPU topology" {
    & nvidia-smi topo -m
}

Add-Section "Filesystem capacity" {
    Get-PSDrive -PSProvider FileSystem |
        Select-Object Name, Used, Free, Root |
        Format-Table -AutoSize
}

Add-Section "WSL status" {
    & wsl --status
}

Add-Section "WSL distributions" {
    & wsl --list --verbose
}

Add-Section "Docker version" {
    & docker version
}

Add-Section "Python version" {
    & python --version
}

Add-Section "Git version" {
    & git --version
}

$report | Set-Content -Encoding UTF8 -LiteralPath $outputFile
Write-Output "Environment report written to: $outputFile"

