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
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $false
}

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

$script:AzCli = Get-AzCliCommand

function Get-AzCliJson {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $result = Invoke-AzCliRaw -Arguments $Arguments
    if ($result.ExitCode -ne 0) {
        throw "Azure CLI command failed: az $($Arguments -join ' ')`n$($result.Text)"
    }
    if ([string]::IsNullOrWhiteSpace(($result.Stdout -join ""))) {
        return $null
    }
    return ($result.Stdout -join "`n") | ConvertFrom-Json
}

function Invoke-AzCliRaw {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $stdoutPath = Join-Path $env:TEMP ("az-{0}.stdout" -f [guid]::NewGuid().ToString("N"))
    $stderrPath = Join-Path $env:TEMP ("az-{0}.stderr" -f [guid]::NewGuid().ToString("N"))

    try {
        $process = Start-Process -FilePath $script:AzCli -ArgumentList $Arguments -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        $stdoutLines = if (Test-Path $stdoutPath) { Get-Content -Path $stdoutPath -ErrorAction SilentlyContinue } else { @() }
        $stderrLines = if (Test-Path $stderrPath) { Get-Content -Path $stderrPath -ErrorAction SilentlyContinue } else { @() }

        return [pscustomobject]@{
            ExitCode = $process.ExitCode
            Stdout = @($stdoutLines)
            Stderr = @($stderrLines)
            Text = ((@($stdoutLines) + @($stderrLines)) -join "`n")
        }
    }
    finally {
        if (Test-Path $stdoutPath) {
            Remove-Item -Path $stdoutPath -Force
        }
        if (Test-Path $stderrPath) {
            Remove-Item -Path $stderrPath -Force
        }
    }
}

function Get-UsageEntry {
    param(
        [Parameter(Mandatory = $true)]$UsageItems,
        [AllowEmptyString()][string]$ExactLocalizedValue = "",
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

function Get-AllowedLocationsFromPolicy {
    param([Parameter(Mandatory = $true)][string]$SubscriptionId)

    $assignments = Get-AzCliJson -Arguments @(
        "policy", "assignment", "list",
        "--scope", "/subscriptions/$SubscriptionId",
        "--output", "json"
    )

    foreach ($assignment in @($assignments)) {
        if ($assignment.displayName -eq "Allowed resource deployment regions") {
            return @($assignment.parameters.listOfAllowedLocations.value)
        }
    }

    return @()
}

function Get-EffectiveLocations {
    param(
        [Parameter(Mandatory = $true)][string[]]$Preferred,
        [Parameter(Mandatory = $true)][string[]]$Allowed
    )

    if (@($Allowed).Count -eq 0) {
        return $Preferred
    }

    $effective = [System.Collections.Generic.List[string]]::new()

    foreach ($location in $Preferred) {
        if ($Allowed -contains $location -and -not $effective.Contains($location)) {
            $effective.Add($location)
        }
    }

    foreach ($location in $Allowed) {
        if (-not $effective.Contains($location)) {
            $effective.Add($location)
        }
    }

    return $effective.ToArray()
}

function Test-SkuAvailable {
    param(
        [Parameter(Mandatory = $true)][string]$Location,
        [Parameter(Mandatory = $true)][string]$SkuName
    )

    $skuList = Get-AzCliJson -Arguments @(
        "vm", "list-skus",
        "--location", $Location,
        "--size", $SkuName,
        "--resource-type", "virtualMachines",
        "--all",
        "--output", "json"
    )

    $sku = $null
    foreach ($item in @($skuList)) {
        if ($item.name -eq $SkuName) {
            $sku = $item
            break
        }
    }

    if ($null -eq $sku) {
        return $false
    }

    if ($null -eq $sku.restrictions) {
        return $true
    }

    return @($sku.restrictions).Count -eq 0
}

function Get-PlacementCandidates {
    param(
        [Parameter(Mandatory = $true)][string[]]$Locations,
        [Parameter(Mandatory = $true)][string[]]$Sizes
    )

    $candidates = [System.Collections.Generic.List[object]]::new()

    foreach ($location in $Locations) {
        Write-Host "Checking quotas for $location..."
        $usage = Get-AzCliJson -Arguments @("vm", "list-usage", "--location", $location, "--output", "json")
        $regionalEntry = Get-UsageEntry -UsageItems $usage -ExactLocalizedValue "Total Regional vCPUs"
        $t4FamilyEntry = Get-UsageEntry -UsageItems $usage -ExactLocalizedValue "" -LooseMatch "NC.*T4.*Family"
        $quotaInfoAvailable = ($null -ne $regionalEntry) -and ($null -ne $t4FamilyEntry)

        if (-not $quotaInfoAvailable) {
            Write-Warning "Quota information is unavailable for $location. Falling back to SKU probing and VM create."
        }

        foreach ($size in $Sizes) {
            $requiredVCpu = Get-RequiredVCpuCount -SkuName $size
            $regionalAvailable = $null
            $familyAvailable = $null
            $regionalLimit = $null
            $familyLimit = $null

            if ($quotaInfoAvailable) {
                $regionalAvailable = [int]$regionalEntry.limit - [int]$regionalEntry.currentValue
                $familyAvailable = [int]$t4FamilyEntry.limit - [int]$t4FamilyEntry.currentValue
                $regionalLimit = [int]$regionalEntry.limit
                $familyLimit = [int]$t4FamilyEntry.limit

                if ($regionalAvailable -lt $requiredVCpu) {
                    Write-Warning "Skipping $size in $location because only $regionalAvailable regional vCPUs are available."
                    continue
                }
                if ($familyAvailable -lt $requiredVCpu) {
                    Write-Warning "Skipping $size in $location because only $familyAvailable NC/T4 family vCPUs are available."
                    continue
                }
            }

            Write-Host "Checking SKU availability for $size in $location..."
            if (Test-SkuAvailable -Location $location -SkuName $size) {
                $candidates.Add([pscustomobject]@{
                    Location = $location
                    Size = $size
                    RequiredVCpu = $requiredVCpu
                    QuotaInfoAvailable = $quotaInfoAvailable
                    RegionalLimit = $regionalLimit
                    RegionalAvailable = $regionalAvailable
                    FamilyLimit = $familyLimit
                    FamilyAvailable = $familyAvailable
                })
            }
        }
    }

    if ($candidates.Count -eq 0) {
        throw "No usable region/SKU combination was found in the configured priority list."
    }

    return $candidates
}

if (-not $script:AzCli) {
    throw "Azure CLI is not installed on this machine. Install Azure CLI, run 'az login', and rerun this script."
}

if ($SubscriptionId) {
    & $script:AzCli account set --subscription $SubscriptionId | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to set the requested subscription."
    }
}

$account = Get-AzCliJson -Arguments @("account", "show", "--output", "json")
$allowedLocations = Get-AllowedLocationsFromPolicy -SubscriptionId $account.id
$effectiveLocations = Get-EffectiveLocations -Preferred $LocationPreference -Allowed $allowedLocations

if (@($allowedLocations).Count -gt 0) {
    Write-Host "Subscription-allowed deployment regions:"
    foreach ($location in $allowedLocations) {
        Write-Host " - $location"
    }
}

$placements = Get-PlacementCandidates -Locations $effectiveLocations -Sizes $SizePreference

Write-Host "Placement candidates:"
foreach ($candidate in $placements) {
    Write-Host " - $($candidate.Location) / $($candidate.Size)"
}

$templatePath = Join-Path $PSScriptRoot "cloud-init.yaml.tmpl"
$customDataPath = Join-Path $env:TEMP "$VmName-cloud-init.yaml"
$template = Get-Content -Raw -Path $templatePath
$customData = $template.Replace("__ADMIN_USERNAME__", $AdminUsername)
$customData = $customData.Replace("__REPO_URL__", $RepoUrl)
$customData = $customData.Replace("__BRANCH__", $Branch)
$customData = $customData.Replace("__REPO_DIR__", $RepoDir)
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($customDataPath, $customData, $utf8NoBom)

try {
    $resourceGroupLocation = $placements[0].Location
    & $script:AzCli group create --name $ResourceGroupName --location $resourceGroupLocation --output json | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create or refresh resource group $ResourceGroupName."
    }

    $placement = $null
    $vm = $null
    $lastFailure = $null

    foreach ($candidate in $placements) {
        Write-Host "Attempting VM create in $($candidate.Location) with $($candidate.Size)..."
        $createResult = Invoke-AzCliRaw -Arguments @(
            "vm", "create",
            "--resource-group", $ResourceGroupName,
            "--name", $VmName,
            "--location", $candidate.Location,
            "--image", "Ubuntu2204",
            "--size", $candidate.Size,
            "--admin-username", $AdminUsername,
            "--authentication-type", "ssh",
            "--generate-ssh-keys",
            "--public-ip-sku", "Standard",
            "--storage-sku", "Premium_LRS",
            "--custom-data", $customDataPath,
            "--output", "json"
        )

        if ($createResult.ExitCode -eq 0) {
            $placement = $candidate
            $vm = $createResult.Text | ConvertFrom-Json
            break
        }

        $lastFailure = $createResult.Text
        Write-Warning "VM create failed for $($candidate.Location) / $($candidate.Size). Trying the next candidate."
    }

    if ($null -eq $vm -or $null -eq $placement) {
        throw "VM creation failed for every candidate. Last Azure CLI output:`n$lastFailure"
    }

    & $script:AzCli vm auto-shutdown --resource-group $ResourceGroupName --name $VmName --time $AutoShutdownTime --timezone $AutoShutdownTimezone | Out-Null
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
