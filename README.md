# 使用方案

## 烧入单片机代码

## 使用py命令

本项目已加入上一级 `3D-VRO` 的 uv 工作区，统一使用
`3D-VRO/.venv` 和根目录的 `uv.lock`。旧的子目录 `.venv` 不再用于 uv 工作区运行。
以下命令均在 `W:\work\3D-VRO` 根目录执行。

日常通过 `uv run --no-sync python` 使用已配置的根环境，避免启动时同步整个项目。
首次安装方式见根目录 README 的“平台运动：统一使用根目录环境”。

先预览三级海况（吴淞号）的角度计算，不发送串口指令：

```powershell
uv run --no-sync python Stewart_Platform_Motion_Simulator/play_6dof_to_platform.py --csv Stewart_Platform_Motion_Simulator/motion_capture/sea_state_3_WuSong/realistic_ship_6dof.csv --seconds 10 --translation-scale 0 --rotation-scale 1 --playback-speed 1
```

向平台发送运动指令（将 `COM4` 替换为实际串口）：

```powershell
uv run --no-sync python Stewart_Platform_Motion_Simulator/play_6dof_to_platform.py --csv Stewart_Platform_Motion_Simulator/motion_capture/sea_state_3_WuSong/realistic_ship_6dof.csv --port COM4 --seconds 10 --translation-scale 0 --rotation-scale 1 --playback-speed 1 --send
```
