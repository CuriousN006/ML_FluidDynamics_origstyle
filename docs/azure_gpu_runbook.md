# Azure GPU Runbook

This runbook keeps the existing local projects unchanged and uses a separate
Azure-targeted sibling clone rooted at `D:\Projects\ML_FluidDynamics_azure_gpu`.

## Defaults

- Branch: `codex/autoresearch/20260311-fluid-rtx3070-latent-sweep`
- Resource group: `rg-mlfd-azure-gpu-kr`
- VM name: `vm-mlfd-t4-01`
- VM size priority: `Standard_NC4as_T4_v3`, then `Standard_NC8as_T4_v3`
- Region priority: `koreacentral`, `eastus`, `westeurope`
- OS: `Ubuntu2204`
- Admin username: `azureuser`
- Auto-shutdown: `01:00` in `Korea Standard Time`

## One-Time Local Prerequisites

1. Install Azure CLI if `az` is missing on this machine.
2. Run `az login`.
3. Optionally set the active subscription with:

```powershell
az account set --subscription "<subscription-id>"
```

4. Check local prerequisites:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\azure\check_prereqs.ps1
```

## Provision The VM

The provisioning script checks:

- `Total Regional vCPUs`
- NC/T4 family quota
- requested SKU availability in each priority region
- subscription policy assignments for `Allowed resource deployment regions`

Then it creates the resource group, provisions the VM, and configures
auto-shutdown.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\azure\provision_vm.ps1
```

If `koreacentral` cannot host the VM because of quota or SKU restrictions, the
script automatically retries `eastus` and then `westeurope`.

If the subscription has an `Allowed resource deployment regions` policy
assignment, the script automatically intersects the preferred region order with
that allowed-region list before it starts probing SKUs or creating the VM.

## Upload The Dataset

`CYLINDER_ALL.mat` is intentionally not tracked by git. After the VM is created,
copy it into the repo root on the VM:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\azure\upload_data.ps1 `
  -LocalDataFile "D:\Projects\ML_FluidDynamics\CYLINDER_ALL.mat"
```

## Bootstrap And Validate

The remote checks script runs the following sequence over SSH:

1. `scripts/azure/bootstrap_repo.sh`
2. `python -m mlfd.setup`
3. `python -m mlfd.run_nonlinear --smoke`
4. `python -m mlfd.run_nonlinear ... --output-tag azure-baseline`

```powershell
powershell -ExecutionPolicy Bypass -File scripts\azure\run_remote_checks.ps1
```

## Baseline Command

The Azure baseline uses the current `exp-0051` reference configuration:

```bash
python -m mlfd.run_nonlinear \
  --output-tag azure-baseline \
  --layout portrait \
  --ae-architecture residual_refine \
  --ae-width-mult 1.0 \
  --coordconv false \
  --coarse-loss-weight 0.25 \
  --coarse-blur-kernel 9 \
  --coarse-blur-sigma 2.0 \
  --refine-blocks 1 \
  --refine-channels-mult 1.25 \
  --dynamics-model residual_linear \
  --latent-dim 32 \
  --gradient-loss-weight 0.10 \
  --fft-loss-weight 0.0 \
  --latent-l1-weight 1e-4 \
  --dyn-l2-weight 0 \
  --rollout-loss-weight 0.15 \
  --train-rollout-horizon 16 \
  --train-rollout-stride 8 \
  --validation-rollout-horizon 24 \
  --validation-rollout-stride 4 \
  --deterministic true
```

## Daily Operating Rule

- Stop and deallocate the VM after each experiment session.
- Keep `output/` and `CYLINDER_ALL.mat` untracked.
- Commit and push only code and documentation changes.
