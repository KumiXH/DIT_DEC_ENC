param(
    [Parameter(Mandatory = $true)]
    [string]$PythonExe,
    [string]$OutputRoot = "work/smoke"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$outputPath = Join-Path $projectRoot $OutputRoot
$dataPath = Join-Path $outputPath "data"
$runsPath = Join-Path $outputPath "runs"

& $PythonExe -m distill_codec.cli make-mock-data --output $dataPath --count 4 --size 32

$configs = @(
    "wan_encoder.yaml",
    "wan_decoder.yaml",
    "flashvsr_lq_proj.yaml",
    "flashvsr_decoder_unconditional.yaml",
    "flashvsr_decoder_conditional.yaml"
)

foreach ($configName in $configs) {
    $configPath = Join-Path $projectRoot "configs/smoke/$configName"
    $runName = [IO.Path]::GetFileNameWithoutExtension($configName)
    $runPath = Join-Path $runsPath $runName
    & $PythonExe -m distill_codec.cli train `
        --config $configPath `
        --set "data.lq_root=$($dataPath)/lq" `
        --set "data.gt_root=$($dataPath)/gt" `
        --set "data.lq_size=[32,32]" `
        --set "data.gt_size=[32,32]" `
        --set "run.output_dir=$runPath" `
        --set "trainer.max_steps=1"
}

