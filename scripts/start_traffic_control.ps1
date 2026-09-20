param([switch]$NoBrowser)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$webRoot = Join-Path $projectRoot "web"
$artifactRoot = Join-Path $projectRoot "api\artifacts"
$logRoot = Join-Path $artifactRoot "logs"
$runtimePath = Join-Path $artifactRoot "launcher-runtime.json"
$networkPath = Join-Path $projectRoot "model\sumo\intersection.net.xml"
$trainingUrl = "http://127.0.0.1:5173/#/training"
$apiProcess = $null
$webProcess = $null

function Test-HttpReady {
    param([Parameter(Mandatory = $true)][string]$Uri)

    try {
        $response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 2
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 500
    }
    catch {
        return $false
    }
}

function Wait-ForHttp {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$Name,
        [int]$TimeoutSeconds = 30
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-HttpReady -Uri $Uri) {
            return
        }
        Start-Sleep -Milliseconds 500
    }
    throw "$Name 준비 시간이 초과되었습니다. $logRoot 폴더의 로그를 확인하세요."
}

function Get-StartTicks {
    param([System.Diagnostics.Process]$Process)
    return $Process.StartTime.ToUniversalTime().Ticks.ToString()
}

try {
    New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

    $venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        $python = $venvPython
    }
    else {
        $python = (Get-Command python.exe -ErrorAction Stop).Source
    }

    $node = (Get-Command node.exe -ErrorAction Stop).Source
    $viteEntry = Join-Path $webRoot "node_modules\vite\bin\vite.js"
    if (-not (Test-Path -LiteralPath $viteEntry)) {
        throw "웹 패키지가 설치되지 않았습니다. web 폴더에서 npm install을 먼저 실행하세요."
    }

    if (-not (Test-Path -LiteralPath $networkPath)) {
        Write-Host "[1/4] SUMO 교차로 네트워크를 생성합니다..." -ForegroundColor Cyan
        & $python -m model.sumo.build_network
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $networkPath)) {
            throw "SUMO 네트워크 생성에 실패했습니다. SUMO 설치 상태를 확인하세요."
        }
    }
    else {
        Write-Host "[1/4] SUMO 네트워크 준비 완료" -ForegroundColor DarkGray
    }

    if (Test-HttpReady -Uri "http://127.0.0.1:8000/api/health") {
        Write-Host "[2/4] 실행 중인 API 서버를 사용합니다." -ForegroundColor DarkGray
    }
    else {
        Write-Host "[2/4] API 서버를 시작합니다..." -ForegroundColor Cyan
        $apiProcess = Start-Process `
            -FilePath $python `
            -ArgumentList @("-m", "api.app.main") `
            -WorkingDirectory $projectRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $logRoot "api.stdout.log") `
            -RedirectStandardError (Join-Path $logRoot "api.stderr.log") `
            -PassThru
        Wait-ForHttp -Uri "http://127.0.0.1:8000/api/health" -Name "API 서버"
    }

    if (Test-HttpReady -Uri "http://127.0.0.1:5173") {
        Write-Host "[3/4] 실행 중인 웹 서버를 사용합니다." -ForegroundColor DarkGray
    }
    else {
        Write-Host "[3/4] 웹 서버를 시작합니다..." -ForegroundColor Cyan
        $webProcess = Start-Process `
            -FilePath $node `
            -ArgumentList @("`"$viteEntry`"", "--host", "127.0.0.1") `
            -WorkingDirectory $webRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $logRoot "web.stdout.log") `
            -RedirectStandardError (Join-Path $logRoot "web.stderr.log") `
            -PassThru
        Wait-ForHttp -Uri "http://127.0.0.1:5173" -Name "웹 서버"
    }

    if ($null -ne $apiProcess -or $null -ne $webProcess) {
        $runtime = [ordered]@{
            created_at = (Get-Date).ToUniversalTime().ToString("o")
            api_pid = if ($null -ne $apiProcess) { $apiProcess.Id } else { $null }
            api_start_ticks = if ($null -ne $apiProcess) { Get-StartTicks -Process $apiProcess } else { $null }
            web_pid = if ($null -ne $webProcess) { $webProcess.Id } else { $null }
            web_start_ticks = if ($null -ne $webProcess) { Get-StartTicks -Process $webProcess } else { $null }
        }
        $runtime | ConvertTo-Json | Set-Content -LiteralPath $runtimePath -Encoding UTF8
    }

    if ($NoBrowser) {
        Write-Host "[4/4] 브라우저 자동 열기를 생략했습니다." -ForegroundColor DarkGray
    }
    else {
        Write-Host "[4/4] 브라우저에서 학습실을 엽니다." -ForegroundColor Green
        Start-Process $trainingUrl
    }
    Write-Host "Traffic Control Lab 실행 완료" -ForegroundColor Green
    Write-Host "종료할 때는 stop_traffic_control.bat을 실행하세요." -ForegroundColor DarkGray
    exit 0
}
catch {
    foreach ($startedProcess in @($webProcess, $apiProcess)) {
        if ($null -ne $startedProcess -and -not $startedProcess.HasExited) {
            Stop-Process -Id $startedProcess.Id -ErrorAction SilentlyContinue
        }
    }
    Write-Host "Traffic Control Lab을 실행하지 못했습니다." -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
