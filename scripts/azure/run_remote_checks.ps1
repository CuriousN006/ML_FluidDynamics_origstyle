[CmdletBinding()]
param(
    [string]$ResourceGroupName = "rg-mlfd-azure-gpu-kr",
    [string]$VmName = "vm-mlfd-t4-01",
    [string]$AdminUsername = "azureuser",
    [string]$RepoDir = "/home/azureuser/ML_FluidDynamics_azure_gpu",
    [switch]$SkipBaseline
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

function Invoke-Remote {
    param([Parameter(Mandatory = $true)][string]$PublicIp, [Parameter(Mandatory = $true)][string]$CommandText)

    & ssh -o StrictHostKeyChecking=accept-new "$AdminUsername@$PublicIp" $CommandText
    if ($LASTEXITCODE -ne 0) {
        throw "Remote command failed: $CommandText"
    }
}

$azCli = Get-AzCliCommand

if (-not $azCli) {
    throw "Azure CLI is not installed."
}
if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    throw "ssh is not available on this machine."
}

$publicIp = (& $azCli vm show -d --resource-group $ResourceGroupName --name $VmName --query publicIps --output tsv).Trim()
if (-not $publicIp) {
    throw "Could not resolve the VM public IP."
}

$bootstrap = "bash '$RepoDir/scripts/azure/bootstrap_repo.sh' '$RepoDir'"
$setup = "bash -lc 'cd $RepoDir && source .venv/bin/activate && python -m mlfd.setup'"
$smoke = "bash -lc 'cd $RepoDir && source .venv/bin/activate && python -m mlfd.run_nonlinear --smoke --output-tag azure-smoke'"
$baseline = @(
    "bash -lc 'cd $RepoDir && source .venv/bin/activate && python -m mlfd.run_nonlinear",
    "--output-tag azure-baseline",
    "--layout portrait",
    "--ae-architecture residual_refine",
    "--ae-width-mult 1.0",
    "--coordconv false",
    "--coarse-loss-weight 0.25",
    "--coarse-blur-kernel 9",
    "--coarse-blur-sigma 2.0",
    "--refine-blocks 1",
    "--refine-channels-mult 1.25",
    "--dynamics-model residual_linear",
    "--latent-dim 32",
    "--gradient-loss-weight 0.10",
    "--fft-loss-weight 0.0",
    "--latent-l1-weight 1e-4",
    "--dyn-l2-weight 0",
    "--rollout-loss-weight 0.15",
    "--train-rollout-horizon 16",
    "--train-rollout-stride 8",
    "--validation-rollout-horizon 24",
    "--validation-rollout-stride 4",
    "--deterministic true'"
) -join " "

Invoke-Remote -PublicIp $publicIp -CommandText $bootstrap
Invoke-Remote -PublicIp $publicIp -CommandText $setup
Invoke-Remote -PublicIp $publicIp -CommandText $smoke

if (-not $SkipBaseline) {
    Invoke-Remote -PublicIp $publicIp -CommandText $baseline
}

Write-Host "Remote validation completed on $VmName ($publicIp)."
