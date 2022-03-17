# Wiggly: Animating Deformable Objects Using Spacetime Constraints

## Build Instruction
To build the C++ code, you must have cmake and a compiler installed.
For Windows/Visual Studio, you can generate the solution by using cmake-gui

```
cd \path\to\wiggly
mkdir build
cd build
cmake-gui ..
```

Inside cmake-gui: 
1. Click "Add Entry" to add a new STRING value called `CMAKE_PREFIX_PATH` pointing to `/path/to/houdini/toolkit/cmake`.
2. Click "Configure" and make sure platform is x64.
3. Click "Generate" to generate the solution.

Once you have generated the solution, you can go to the build folder to open the solution in Visual Studio.
You should be able to build the solution in Visual Studio and get the dso file right away.
