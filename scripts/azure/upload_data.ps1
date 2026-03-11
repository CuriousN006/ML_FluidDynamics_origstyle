[CmdletBinding()]
param(
    [string]$ResourceGroupName = "rg-mlfd-azure-gpu-kr",
    [string]$VmName = "vm-mlfd-t4-01",
    [string]$AdminUsername = "azureuser",
    [string]$RepoDir = "/home/azureuser/ML_FluidDynamics_azure_gpu",
    [Parameter(Mandatory = $true)][string]$LocalDataFile
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-AzCliCommand {
    $candidates = @(
        "az",
        "az.cmd",
        "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd",
        "C:\Program Files (x86)\Microsoft SDKs\Azure\CLI2\wbin\az.cmd"
    )

    foreach ($candidate in $candidates) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($null -ne $command) {
            return $command.Source
        }
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $null
}

$azCli = Get-AzCliCommand

if (-not (Test-Path $LocalDataFile)) {
    throw "Local data file not found: $LocalDataFile"
}
if (-not $azCli) {
    throw "Azure CLI is not installed."
}
if (-not (Get-Command scp -ErrorAction SilentlyContinue)) {
    throw "scp is not available on this machine."
}

$publicIp = (& $azCli vm show -d --resource-group $ResourceGroupName --name $VmName --query publicIps --output tsv).Trim()
if (-not $publicIp) {
    throw "Could not resolve the VM public IP."
}

$remoteTarget = "{0}@{1}:{2}/CYLINDER_ALL.mat" -f $AdminUsername, $publicIp, $RepoDir
& scp -o StrictHostKeyChecking=accept-new $LocalDataFile $remoteTarget
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upload CYLINDER_ALL.mat to the VM."
}

Write-Host ("Uploaded CYLINDER_ALL.mat to {0}" -f $remoteTarget)
