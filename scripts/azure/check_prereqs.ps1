[CmdletBinding()]
param(
    [string]$SubscriptionId
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

function Test-CommandExists {
    param([Parameter(Mandatory = $true)][string]$Name)

    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

$azCli = Get-AzCliCommand
$report = [ordered]@{
    az_installed     = $null -ne $azCli
    ssh_installed    = Test-CommandExists -Name "ssh"
    scp_installed    = Test-CommandExists -Name "scp"
    azure_logged_in  = $false
    active_account   = $null
    subscription_id  = $null
    tenant_id        = $null
}

if ($report.az_installed) {
    try {
        if ($SubscriptionId) {
            & $azCli account set --subscription $SubscriptionId | Out-Null
        }

        $accountRaw = & $azCli account show --output json 2>$null
        if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace(($accountRaw -join ""))) {
            $account = ($accountRaw -join "`n") | ConvertFrom-Json
            $report.azure_logged_in = $true
            $report.az_path = $azCli
            $report.active_account = $account.name
            $report.subscription_id = $account.id
            $report.tenant_id = $account.tenantId
        }
        else {
            $report.az_path = $azCli
            $report.azure_error = "Azure CLI is installed, but no Azure account is logged in. Run 'az login'."
        }
    }
    catch {
        $report.azure_error = $_.Exception.Message
    }
}
else {
    $report.azure_error = "Azure CLI is not installed."
}

$report | ConvertTo-Json -Depth 4
