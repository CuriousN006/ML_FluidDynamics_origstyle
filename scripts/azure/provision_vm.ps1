[CmdletBinding()]
param(
    [string]$SubscriptionId,
    [string]$ResourceGroupName = "rg-mlfd-azure-gpu-kr",
    [string]$VmName = "vm-mlfd-t4-01",
    [string]$AdminUsername = "azureuser",
    [string]$RepoUrl = "https://github.com/CuriousN006/ML_FluidDynamics.git",
    [string]$Branch = "codex/autoresearch/20260311-fluid-rtx3070-latent-sweep",
    [string]$RepoDir = "/home/azureuser/ML_FluidDynamics_azure_gpu",
    [string[]]$LocationPreference = @("koreacentral", "eastus", "westeurope"),
    [string[]]$SizePreference = @("Standard_NC4as_T4_v3", "Standard_NC8as_T4_v3"),
    [string]$AutoShutdownTime = "0100",
    [string]$AutoShutdownTimezone = "Korea Standard Time"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-AzCliJson {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $raw = & az @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Azure CLI command failed: az $($Arguments -join ' ')"
    }
    if ([string]::IsNullOrWhiteSpace(($raw -join ""))) {
        return $null
    }
    return ($raw -join "`n") | ConvertFrom-Json
}

function Get-UsageEntry {
    param(
        [Parameter(Mandatory = $true)]$UsageItems,
        [Parameter(Mandatory = $true)][string]$ExactLocalizedValue,
        [string]$LooseMatch = ""
    )

    if ($ExactLocalizedValue) {
        foreach ($item in $UsageItems) {
            if ($item.name.localizedValue -eq $ExactLocalizedValue) {
                return $item
            }
        }
    }
    if ($LooseMatch) {
        foreach ($item in $UsageItems) {
            if ($item.name.localizedValue -match $LooseMatch -or $item.name.value -match $LooseMatch) {
                return $item
            }
        }
    }
    return $null
}

function Get-RequiredVCpuCount {
    param([Parameter(Mandatory = $true)][string]$SkuName)

    if ($SkuName -match "NC(\d+)") {
        return [int]$Matches[1]
    }
    throw "Could not infer the vCPU count from SKU name: $SkuName"
}

function Test-SkuAvailable {
    param(
        [Parameter(Mandatory = $true)][string]$Location,
        [Parameter(Mandatory = $true)][string]$SkuName
    )

    $sku = Get-AzCliJson -Arguments @(
        "vm", "list-skus",
        "--location", $Location,
        "--size", $SkuName,
        "--resource-type", "virtualMachines",
        "--all",
        "--query", "[?name=='$SkuName'] | [0]",
        "--output", "json"
    )

    if ($null -eq $sku) {
        return $false
    }

    if ($null -eq $sku.restrictions) {
        return $true
    }

    return @($sku.restrictions).Count -eq 0
}

function Select-TargetPlacement {
    param(
        [Parameter(Mandatory = $true)][string[]]$Locations,
        [Parameter(Mandatory = $true)][string[]]$Sizes
    )

    foreach ($location in $Locations) {
        Write-Host "Checking quotas for $location..."
        $usage = Get-AzCliJson -Arguments @("vm", "list-usage", "--location", $location, "--output", "json")
        $regionalEntry = Get-UsageEntry -UsageItems $usage -ExactLocalizedValue "Total Regional vCPUs"
        $t4FamilyEntry = Get-UsageEntry -UsageItems $usage -ExactLocalizedValue "" -LooseMatch "NC.*T4.*Family"

        if ($null -eq $regionalEntry) {
            Write-Warning "Skipping $location because Total Regional vCPUs quota information is missing."
            continue
        }
        if ($null -eq $t4FamilyEntry) {
            Write-Warning "Skipping $location because NC/T4 family quota information is missing."
            continue
        }

        foreach ($size in $Sizes) {
            $requiredVCpu = Get-RequiredVCpuCount -SkuName $size
            $regionalAvailable = [int]$regionalEntry.limit - [int]$regionalEntry.currentValue
            $familyAvailable = [int]$t4FamilyEntry.limit - [int]$t4FamilyEntry.currentValue

            if ($regionalAvailable -lt $requiredVCpu) {
                Write-Warning "Skipping $size in $location because only $regionalAvailable regional vCPUs are available."
                continue
            }
            if ($familyAvailable -lt $requiredVCpu) {
                Write-Warning "Skipping $size in $location because only $familyAvailable NC/T4 family vCPUs are available."
                continue
            }

            Write-Host "Checking SKU availability for $size in $location..."
            if (Test-SkuAvailable -Location $location -SkuName $size) {
                return [pscustomobject]@{
                    Location = $location
                    Size = $size
                    RequiredVCpu = $requiredVCpu
                    RegionalLimit = [int]$regionalEntry.limit
                    RegionalAvailable = $regionalAvailable
                    FamilyLimit = [int]$t4FamilyEntry.limit
                    FamilyAvailable = $familyAvailable
                }
            }
        }
    }

    throw "No usable region/SKU combination was found in the configured priority list."
}

if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    throw "Azure CLI is not installed on this machine. Install Azure CLI, run 'az login', and rerun this script."
}

if ($SubscriptionId) {
    & az account set --subscription $SubscriptionId | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to set the requested subscription."
    }
}

$null = Get-AzCliJson -Arguments @("account", "show", "--output", "json")
$placement = Select-TargetPlacement -Locations $LocationPreference -Sizes $SizePreference

Write-Host "Selected location: $($placement.Location)"
Write-Host "Selected size: $($placement.Size)"

$templatePath = Join-Path $PSScriptRoot "cloud-init.yaml.tmpl"
$customDataPath = Join-Path $env:TEMP "$VmName-cloud-init.yaml"
$template = Get-Content -Raw -Path $templatePath
$customData = $template.Replace("__ADMIN_USERNAME__", $AdminUsername)
$customData = $customData.Replace("__REPO_URL__", $RepoUrl)
$customData = $customData.Replace("__BRANCH__", $Branch)
$customData = $customData.Replace("__REPO_DIR__", $RepoDir)
Set-Content -Path $customDataPath -Value $customData -Encoding utf8

try {
    & az group create --name $ResourceGroupName --location $placement.Location --output json | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create or refresh resource group $ResourceGroupName."
    }

    $vm = Get-AzCliJson -Arguments @(
        "vm", "create",
        "--resource-group", $ResourceGroupName,
        "--name", $VmName,
        "--location", $placement.Location,
        "--image", "Ubuntu2204",
        "--size", $placement.Size,
        "--admin-username", $AdminUsername,
        "--authentication-type", "ssh",
        "--generate-ssh-keys",
        "--public-ip-sku", "Standard",
        "--storage-sku", "Premium_LRS",
        "--custom-data", $customDataPath,
        "--output", "json"
    )

    & az vm auto-shutdown --resource-group $ResourceGroupName --name $VmName --time $AutoShutdownTime --timezone $AutoShutdownTimezone | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "VM was created, but auto-shutdown configuration did not complete."
    }

    [pscustomobject]@{
        resource_group = $ResourceGroupName
        vm_name = $VmName
        location = $placement.Location
        size = $placement.Size
        required_vcpu = $placement.RequiredVCpu
        regional_vcpu_available = $placement.RegionalAvailable
        family_vcpu_available = $placement.FamilyAvailable
        public_ip = $vm.publicIpAddress
        ssh = "ssh $AdminUsername@$($vm.publicIpAddress)"
        repo_dir = $RepoDir
        auto_shutdown_time = $AutoShutdownTime
        auto_shutdown_timezone = $AutoShutdownTimezone
    } | ConvertTo-Json -Depth 4
}
finally {
    if (Test-Path $customDataPath) {
        Remove-Item -Path $customDataPath -Force
    }
}
