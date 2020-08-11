mkdir build
cd build
cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=../libtorch ..
cmake --build . --config Release
cd ..
py bakkes_patchplugin.py
pause
