# Wiggly: Animating Deformable Objects Using Spacetime Constraints
University of Pennsylvania, CIS 660: Advanced Topics in Computer Graphics

[Wayne Wu](https://www.wuwayne.com/) and Aditya Abhyankar

![Teaser](img/xwalking.png)

## Installation
To use the tools without building, simply copy the `dso`, `config`, `otls` folders to your HOUDINI_USER_PREF_DIR.

## Build Instruction

### Dependencies
* [cmake](https://cmake.org/)
* [dlib](http://dlib.net/)

To build the C++ code, you must install the dependencies listed above and have an appropriate compiler. You should also modify `src/CMakeLists.txt` to point to your Houdini and dlib paths.

Overview
============
This project implements the paper: [Animating deformable objects using sparse spacetime constraints](https://dl.acm.org/doi/10.1145/2601097.2601156) by Schulz et al. in Houdini. As part of the authoring tool, we have designed an intuitive workflow that allows users to easily create the sparse keyframes and generate animation with our custom solver.
