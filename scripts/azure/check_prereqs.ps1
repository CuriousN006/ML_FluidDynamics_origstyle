[CmdletBinding()]
param(
    [string]$SubscriptionId
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-CommandExists {
    param([Parameter(Mandatory = $true)][string]$Name)

    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

$report = [ordered]@{
    az_installed     = Test-CommandExists -Name "az"
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
            az account set --subscription $SubscriptionId | Out-Null
        }

        $account = az account show --output json | ConvertFrom-Json
        $report.azure_logged_in = $true
        $report.active_account = $account.name
        $report.subscription_id = $account.id
        $report.tenant_id = $account.tenantId
    }
    catch {
        $report.azure_error = $_.Exception.Message
    }
}
else {
    $report.azure_error = "Azure CLI is not installed."
}

$report | ConvertTo-Json -Depth 4
