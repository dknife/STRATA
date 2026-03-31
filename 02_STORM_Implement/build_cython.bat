@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64 >nul 2>&1

echo Compiling...
cl /c /nologo /O2 /MD /W3 ^
  /I"C:\Users\young\AppData\Local\Python\pythoncore-3.14-64\Include" ^
  /I"C:\Users\young\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\numpy\_core\include" ^
  /Tc storm\_storm_core.c ^
  /Fo:storm\_storm_core.obj
if errorlevel 1 (echo COMPILE FAILED & exit /b 1)

echo Linking...
link /nologo /DLL /INCREMENTAL:NO ^
  /LIBPATH:"C:\Users\young\AppData\Local\Python\pythoncore-3.14-64\libs" ^
  python314.lib ^
  storm\_storm_core.obj ^
  /OUT:storm\_storm_core.cp314-win_amd64.pyd
if errorlevel 1 (echo LINK FAILED & exit /b 2)

echo BUILD SUCCESS
del storm\_storm_core.obj storm\_storm_core.lib storm\_storm_core.exp 2>nul
