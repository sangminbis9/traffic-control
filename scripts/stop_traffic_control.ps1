$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimePath = Join-Path $projectRoot "api\artifacts\launcher-runtime.json"

function Stop-RecordedProcess {
    param(
        [Nullable[int]]$ProcessId,
        [string]$StartTicks,
        [string]$Name
    )

    if ($null -eq $ProcessId -or [string]::IsNullOrWhiteSpace($StartTicks)) {
        return $false
    }

    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        return $false
    }

    $actualTicks = $process.StartTime.ToUniversalTime().Ticks.ToString()
    if ($actualTicks -ne $StartTicks) {
        Write-Host "$Name PID가 다른 프로세스에 재사용되어 종료하지 않았습니다." -ForegroundColor Yellow
        return $false
    }

    Stop-Process -Id $ProcessId
    $process.WaitForExit(5000) | Out-Null
    Write-Host "$Name 종료 완료" -ForegroundColor Green
    return $true
}

try {
    if (-not (Test-Path -LiteralPath $runtimePath)) {
        Write-Host "실행 파일이 시작한 서버 기록이 없습니다." -ForegroundColor Yellow
        exit 0
    }

    $runtime = Get-Content -LiteralPath $runtimePath -Raw | ConvertFrom-Json
    Stop-RecordedProcess -ProcessId $runtime.web_pid -StartTicks $runtime.web_start_ticks -Name "웹 서버" | Out-Null
    Stop-RecordedProcess -ProcessId $runtime.api_pid -StartTicks $runtime.api_start_ticks -Name "API 서버" | Out-Null
    Remove-Item -LiteralPath $runtimePath -Force
    Write-Host "Traffic Control Lab 서버 정리가 완료되었습니다." -ForegroundColor Green
    exit 0
}
catch {
    Write-Host "서버를 종료하지 못했습니다." -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
