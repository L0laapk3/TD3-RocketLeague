echo off
cls
mkdir build
cd build
cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=../libtorch ..
if %ERRORLEVEL% NEQ 0 GOTO ERROR
cmake --build . --config Release
if %ERRORLEVEL% NEQ 0 GOTO ERROR
cd ..
py bakkes_patchplugin.py
exit

:ERROR
pause
exit