# RAG 一键停止：关闭隧道（含重连窗口）+ 停止全部容器
# 数据全部保留，下次双击 start.bat 即可恢复
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "[停止] 关闭隧道（含自动重连窗口）..." -ForegroundColor Cyan
$targets = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match '7860:127\.0\.0\.1:7860' -and ($_.Name -eq 'ssh.exe' -or $_.Name -eq 'powershell.exe')
}
$parents = $targets | Select-Object -ExpandProperty ParentProcessId -Unique
$targets | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'powershell.exe' -and $_.ProcessId -in $parents } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

Write-Host "[停止] 停止全部容器..." -ForegroundColor Cyan
docker compose -f stage2-milvus/docker-compose.yml stop

Write-Host "[停止] 完成。数据不受影响，双击 start.bat 可恢复。" -ForegroundColor Green
