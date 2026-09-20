param([switch]$SkipTests)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$webRoot = Join-Path $projectRoot "web"
$venvRoot = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][scriptblock]$Command
    )

    Write-Host $Label -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label 단계가 종료 코드 $LASTEXITCODE 로 실패했습니다."
    }
}

try {
    Set-Location $projectRoot

    $systemPython = (Get-Command python.exe -ErrorAction Stop).Source
    $node = (Get-Command node.exe -ErrorAction Stop).Source
    $npm = (Get-Command npm.cmd -ErrorAction Stop).Source

    if (-not (Test-Path -LiteralPath $venvPython)) {
        Invoke-Checked "[1/7] Python 가상환경 생성" { & $systemPython -m venv $venvRoot }
    }
    else {
        Write-Host "[1/7] 기존 Python 가상환경 사용" -ForegroundColor DarkGray
    }

    Invoke-Checked "[2/7] pip 업데이트" { & $venvPython -m pip install --upgrade pip }
    Invoke-Checked "[3/7] 검증된 Python 의존성 설치" { & $venvPython -m pip install -r (Join-Path $projectRoot "requirements-windows.lock.txt") }
    Write-Host "[4/7] 모델/API 의존성 준비 완료" -ForegroundColor DarkGray

    Write-Host "[5/7] 웹 의존성 설치" -ForegroundColor Cyan
    Push-Location $webRoot
    try {
        & $npm ci
        if ($LASTEXITCODE -ne 0) {
            throw "npm ci 단계가 종료 코드 $LASTEXITCODE 로 실패했습니다."
        }
    }
    finally {
        Pop-Location
    }

    Invoke-Checked "[6/7] SUMO 네트워크 생성" { & $venvPython -m model.sumo.build_network }

    if ($SkipTests) {
        Write-Host "[7/7] 검증 생략" -ForegroundColor DarkGray
    }
    else {
        Invoke-Checked "[7/7] 핵심 테스트 실행" { & $venvPython -m pytest model\tests api\tests -q }
        Push-Location $webRoot
        try {
            & $npm test
            if ($LASTEXITCODE -ne 0) {
                throw "웹 테스트가 종료 코드 $LASTEXITCODE 로 실패했습니다."
            }
            & $npm run build
            if ($LASTEXITCODE -ne 0) {
                throw "웹 빌드가 종료 코드 $LASTEXITCODE 로 실패했습니다."
            }
        }
        finally {
            Pop-Location
        }
    }

    Write-Host "설치가 완료되었습니다." -ForegroundColor Green
    Write-Host "이제 start_traffic_control.bat을 더블클릭하세요." -ForegroundColor Green
    Write-Host "Python: $(& $venvPython --version)" -ForegroundColor DarkGray
    Write-Host "Node: $(& $node --version)" -ForegroundColor DarkGray
    exit 0
}
catch {
    Write-Host "Traffic Control Lab 설치에 실패했습니다." -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host "Python, Node.js, Eclipse SUMO가 설치되어 있는지 확인하세요." -ForegroundColor Yellow
    exit 1
}
