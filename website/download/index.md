---
layout: default
title: Download
permalink: /download/
---

{::options parse_block_html="true" /}

## CentOS/RedHat/Fedora users:
Project Chrono release RPMs are available from our [repositories](http://projectchrono.org/repos/projectchrono-repo-0.1.0-1.noarch.rpm).

## Other Users:
Project Chrono should be **cloned** from the public [GIT repository](https://github.com/projectchrono/chrono) and **compiled** from its C++ source code as described below.



Required tools
--------------

### C++ compiler 

Chrono requires a C++11 compliant compiler. The Chrono building process has been tested with Microsoft compilers (required version Visual Studio v.2013 or later), GNU compilers, and Intel compilers.

<div class="ce-info">

In case you do not have a C++ compiler already installed on your computer, you can download one of these free tools:

* Windows 

  * [Microsoft C++, in Visual Studio](https://www.visualstudio.com)  (suggested; the Community edition is also free)
  
  * [MingW GNU C++](http://www.mingw.org/wiki/InstallationHOWTOforMinGW)
  
* Linux 

  * On most distributions, the GNU gcc compiler should be already installed by default. If not, install with:  
	```
	sudo apt-get install gcc build-essential 
	```  
    (assuming you have Ubuntu) 
   
  * or download from [http://gcc.gnu.org](http://gcc.gnu.org)
</div>

<div class="ce-danger">
The initial release of Visual Studio 2015 gives an 
error when compiling Chrono. Using **update 2** of Visual Studio 2015 fixes this problem (download it from 
[this page](https://www.visualstudio.com/en-us/news/vs2015-update2-vs.aspx). )
</div>


### GIT client

The C++ source code of Chrono is hosted on a GIT versioning system. 
You need a GIT client that allows you to clone and pull the most recent release from GIThub. 
There are many alternatives, but we suggest the following:

<div class="well">
<h4> <span class="glyphicon glyphicon-download-alt"></span> Download a GIT client</h4>
<a href="https://www.sourcetreeapp.com/">Download SourceTree (for Windows and OS X)</a>
</div>
	
### CMake

The CMake tool is **required** in order to perform the building of the
Chrono libraries and demos. It is cross-platform and available
for many operating systems (choose the one that you need, ex. Windows,
Linux, etc.):

<div class="well">
<h4> <span class="glyphicon glyphicon-download-alt"></span> Download CMake</h4>
<a href="http://www.cmake.org/cmake/resources/software.html">Download CMake from the www.cmake.org web site.</a>
</div>

### Irrlicht 3D

The [Irrlicht](http://irrlicht.sourceforge.net/downloads.html) library
is **required** in order to compile the Chrono demos that use a
3D visualization. It is a cross-platform 3D visualization library, based
on OpenGL and DirectX.

<div class="well">
<h4> <span class="glyphicon glyphicon-download-alt"></span> Download Irrlicht</h4>
<a href="http://downloads.sourceforge.net/irrlicht/irrlicht-1.8.2.zip">Download Irrlicht v.1.8.2 </a>
</div>


{:info: .ce-info}
The new release v.1.8.2 of Irrlicht is tested to be stable and working well with Chrono.  
Release v.1.8.3 does not contain the precompiled 64bit dlls.  
Release v.1.8.0 has some issues with soft shadows.
{:info}

{:warning: .ce-warning}
You cannot link 64bit Chrono libraries and 32bit Irrlicht libraries, or viceversa. If you decide to build Chrono in 64bit (suggested), you must link 64 bit Irrlicht .lib and .dll. 
{:warning}


Optional tools
--------------

### Matlab

The *optional* **MATLAB module** in Chrono provides the support
for interoperation with Matlab(TM). If you want to build that unit and
to test its demos, you must have the Matlab API installed in your
computer: that is distributed in the main Matlab software, so if you
already have Matlab installed you are fine. If not, just do not compile and use the
MATLAB module in the Chrono project. If you want to buy the
Matlab software, go to
[<http://www.mathworks.com>](http://www.mathworks.com).

### MPI

The *optional* **MPI module** in Chrono provides functionality
for performing distributed parallel computing on large clusters of
computers, using the Message Passing Interface (MPI) standard. If you
want to build that unit and to test its demos, you must have the MPICH 2
installed in your computer. It can be downloaded from [this
site](http://www.mcs.anl.gov/research/projects/mpich2/downloads/index.php?s=downloads).

### CUDA

The *optional* **PARALLEL module** in Chrono provides functionality
for performing parallel computation on NVIDIA GPU boards. If you want to
build that unit and to test its demos, you must have the CUDA SDK and
toolkit installed in your computer. They can be downloaded from [this
site](http://developer.nvidia.com/object/cuda_downloads.html).

### POVray

The *optional* **POSTPROCESSING module** in Chrono provides
functionality for exporting simulation data in formats that can be
rendered by external software. No libraries are needed to build or use
this unit; however you may need raytracing software for
processing the saved data. Currently, the POVray 3D raytracer is
supported and tested. Download it from [this
site](http://www.povray.org).

### GNUplot

The *optional* **POSTPROCESSING module** in Chrono provides
functionality for exporting simulation data. One of the output file formats 
is for plotting graphs with the free GNUplot tool. Download it from [this
site](http://www.gnuplot.info).




---


Chrono::PyEngine
----------------

This is the ready-to-use Chrono::PyEngine for Python. It is a module
that can be installed in your Python environment so that you can access
the functions of Chrono via the [Python
language](http://www.python.org). This can be an easier way to use
Chrono because Python is more friendly than C++ (but remember
that not all C++ functions are mapped to Python, moreover Python is
slower than C++).

The installer automatically detects your Python installation and adds
the Chrono module.

Note! currently only Python versions 3.2, 3.3 are supported. Also, only
the 32 bit version of Python is currently supported with this installer.


<div class="well">
<h4> <span class="glyphicon glyphicon-download-alt"></span> Download PyChrono for Python</h4>

<a href="http://www.chronoengine.info/download/PyChronoEngine_v1.8.0.exe"> Download Chrono::PyEngine v.1.8.0 module for Python v.3.3 64bit</a><br />
<a href="http://www.chronoengine.info/download/PyChronoEngine_v1.7.1.exe"> Download Chrono::PyEngine v.1.7.1 module for Python v.3.3 32bit</a><br />
<a href="http://www.chronoengine.info/download/PyChronoEngine_v1.7.0.exe"> Download Chrono::PyEngine v.1.7.0 module for Python v.3.2 32bit</a><br />

<h5> older versions:</h5>

<a href="http://www.chronoengine.info/download/ChronoEngine_for_Python_v1.61.exe"> Download Chrono::PyEngine v.1.6.1 module for Python v.3.2 32bit</a><br />
<a href="http://www.chronoengine.info/download/ChronoEngine_for_Python_v1.60.exe"> Download Chrono::PyEngine v.1.6.0 module for Python v.3.2 32bit</a><br />
<a href="http://www.chronoengine.info/download/ChronoEngine_for_Python_v1.50.exe"> Download Chrono::PyEngine v.1.5.0 module for Python v.3.2 32bit</a>

</div>

{:info: ce-info} 
If you plan to use ''only'' Python, this is the only installer that you need to download.
{:info}



{:info: ce-info}
If you need to plot graphs in Python, we suggest you to consider also the installation of both 
[MatplotLib](http://matplotlib.org/downloads.html) and 
[Numpy](http://www.numpy.org) 
Python packages.
{:info}

---


Chrono::SolidWorks
------------------

This optional tool can be installed as an add-in for the
[SolidWorks](http://www.SolidWorks.com) 3D CAD, that is widely used by
many engineering companies. After you install this add-in, you will find
a new exporter tool in the right panel of SolidWorks: this can be used
to generate .py files with PyChrono scene descriptions
containing masses, constraints, etc.

The installer automatically detects your SolidWorks installation (v.2011
and v.2012 64bit supported and tested, for the moment) and adds the
Chrono add-in.

		  
<div class="well">
<h4> <span class="glyphicon glyphicon-download-alt"></span> Download Chrono::SolidWorks add-in for SolidWorks</h4>
<a href="http://www.chronoengine.info/download/ChronoEngine_SolidWorks_v2.04.exe"> Download Chrono::SolidWorks v.2.0.4 add-in for SolidWorks 2013 and higher (beta)</a><br />

<h5> older versions:</h5>

<a href="http://www.chronoengine.info/download/ChronoEngine_SolidWorks_v2.01.exe"> Download Chrono::SolidWorks v.2.0.1 add-in for SolidWorks 2012</a><br />
<a href="http://www.chronoengine.info/download/ChronoEngine_SolidWorks_v1.70.exe"> Download Chrono::SolidWorks v.1.7.0 add-in for SolidWorks 2011</a><br />
<a href="http://www.chronoengine.info/download/ChronoEngine_SolidWorks_v1.62.exe"> Download Chrono::SolidWorks v.1.6.2 add-in for SolidWorks 2011</a>

</div>
