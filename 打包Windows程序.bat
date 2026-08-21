@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo 未找到项目虚拟环境 .venv。
  echo 请先创建虚拟环境并安装 requirements.txt 和 requirements-build.txt。
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
  echo 当前 .venv 尚未安装 PyInstaller。
  echo 请运行：.venv\Scripts\python.exe -m pip install -r requirements-build.txt
  pause
  exit /b 1
)

echo 正在生成 Windows 单目录程序，请稍候...
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean chip_matching_gui.spec
if errorlevel 1 (
  echo 打包失败，请查看上方错误信息。
  pause
  exit /b 1
)

copy /Y "试用说明.txt" "dist\芯片替代料匹配工具\试用说明.txt" >nul

echo.
echo 打包完成：dist\芯片替代料匹配工具
echo 对外发送时请压缩并发送整个文件夹，不要只发送 EXE。
pause
