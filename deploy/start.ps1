# RAG 一键启动：本地全栈容器 + 云端隧道（断线自动重连）
# 用法：双击 deploy\start.bat（或右键"使用 PowerShell 运行"本文件）
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Say($msg) { Write-Host "[启动] $msg" -ForegroundColor Cyan }

# 1) 确保 Docker 引擎就绪
Say "检查 Docker 引擎..."
docker info *> $null
if (-not $?) {
    Say "Docker Desktop 未运行，尝试自动启动（最多等 2 分钟）..."
    $dd = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dd) { Start-Process $dd }
    $ok = $false
    foreach ($i in 1..60) {
        Start-Sleep -Seconds 2
        docker info *> $null
        if ($?) { $ok = $true; break }
    }
    if (-not $ok) { Say "Docker 引擎未就绪，请手动打开 Docker Desktop 后重试"; exit 1 }
}
Say "Docker 引擎就绪。"

# 2) 起全部容器（Milvus / ES / 三个应用服务；首次或代码变更时自动构建镜像）
Say "启动容器（首次构建可能多等几分钟）..."
docker compose -f stage2-milvus/docker-compose.yml up -d --build

# 3) 等本地界面就绪
Say "等待应用服务就绪..."
$ready = $false
foreach ($i in 1..60) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:7860/" -TimeoutSec 3 -UseBasicParsing
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch {}
    Start-Sleep -Seconds 2
}
if (-not $ready) {
    Say "等待超时——查看日志: docker compose -f stage2-milvus/docker-compose.yml logs web"
    exit 1
}
Say "本地界面就绪: http://127.0.0.1:7860"

# 4) 启动云端隧道（独立窗口；断线自动重连，关掉该窗口 = 网站下线）
Say "启动云端隧道（会弹出一个新窗口）..."
$tunnelCmd = '$Host.UI.RawUI.WindowTitle="RAG 隧道（关闭=下线）"; while($true){ ssh -N -R 7860:127.0.0.1:7860 -o ExitOnForwardFailure=yes -o BatchMode=yes -o LogLevel=ERROR rag; Write-Host "隧道断开，5 秒后重连..." -ForegroundColor Yellow; Start-Sleep -Seconds 5 }'
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($tunnelCmd))
Start-Process powershell -ArgumentList '-NoExit','-NoProfile','-EncodedCommand',$encoded

# 5) 从云端反向验证整条链路
Say "从云端验证隧道..."
$cloudOk = $false
foreach ($i in 1..20) {
    Start-Sleep -Seconds 2
    $code = (ssh rag "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:7860/") 2>$null
    if ($code -match "200") { $cloudOk = $true; break }
}
Write-Host ""
if ($cloudOk) {
    Say "全部就绪 ✔"
    Write-Host "  本地界面: http://127.0.0.1:7860" -ForegroundColor Green
    Write-Host "  公网入口: http://8.160.116.153:8080 （用户名 rag）" -ForegroundColor Green
} else {
    Say "隧道可能还在建立——稍等片刻刷新页面，或查看那个隧道窗口的状态"
}
