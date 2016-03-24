Chrono Frequently Asked Questions {#faq_root}
==========================


#### Who owns Chrono?
Chrono is owned and copyrighted by Project::Chrono, a non-profit set up in the US. 
<br><br>

#### Is Chrono free?
The entire Chrono software infrastructure is open source and released under a permissive [BSD3](https://opensource.org/licenses/BSD-3-Clause) license. As such, one can use, modify, redistribute, sell, etc., the software. 
<br><br>

#### Does Chrono have a graphical user interface?
No, it doesn't. However, if you need a CAD-like interface, consider the use of the 
[Chrono::SolidWorks](@ref tutorial_table_of_content_chrono_solidworks) plug-in. It supports a work flow in which you can develop an assembly in SolidWorks&copy; and then export it as a model that can be subsequently imported and simulated in Chrono. Another alternative is the use of a third party [commercial tool](http://www.simlab-soft.com/technologies/simlab-simulation.aspx) that embeds Chrono and which provides a GUI.
<br><br>

#### What sort of problems is Chrono positioned to solve?
Chrono is a simulation tool that can be used to understand how physical systems change their state in time. This is a broad description, but this is where Chrono is heading. Today, Chrono provides really good support if your physical system is a mechanical system such as a vehicle, robot, suspension system, etc. It provides good support if you are interested in the motion of compliant systems, such as collections of beams, shells, etc. This support comes via Chrono::FEA. Finally, there is fledgling support for fluid-solid interaction and fluid dynamics, which comes via Chrono::FSI.
<br><br>

#### What applications has Chrono been used for?
Vehicle dynamics (wheeled and tracked), terramechanics, robotics, granular dynamics, mechanism design, oil well capping, farming, food processing, building collapse, connected autonomous vehicles.
<br><br>

#### Why shouldn't I use Bullet@copy, or ODE@copy, or Physx@copy? They too look at how systems change in time and for some problems they do it many times faster than Chrono.
Chrono takes a physics-based approach to the modeling and simulation of complex dynamic systems.  As such, we are paying a price for trying to stay faithful to the underlying physics governing the time evolution of the system of interest. We are committing a significant chunk of our effort to validate the modeling and simulation techniques we rely upon in Chrono. Many of these validation activities are reported in several [technical reports and theses](http://sbel.wisc.edu/Publications/).
<br><br>

#### I want to use Chrono. Do I have to be a good programmer?
No. You only need to be able to install the software and go through a couple of [examples/tutorials](@ref tutorial_root) to get the gist of it. Moreover, if you need a jump start to your project you might want to take a look at the [Chrono model repository](@ref model_root). If you are lucky, you'll find a model close to what you need, in which case you'll get a jump start to your project. Finally, there is also a [consulting](http://projectchrono.org/consulting/) avenue.
<br><br>

#### How can I contribute?
You can contribute in many ways:
- If you want to add to the Chrono software infrastructure, make a pull request in [GitHub](https://github.com/projectchrono/chrono)
- If you put together a model and want to make it part of the [Chrono model repository](@ref model_root) you can make a pull request in [GitHub](https://github.com/projectchrono/chrono)
- If you put together a tutorial or generated a well documented Chrono example, follow the same [GitHub](https://github.com/projectchrono/chrono) pull request path
- If you feel like Chrono is the best thing since slice bread and you want to support its development, make a tax deductible donation to the University of Wisconsin-Madison or University of Parma, Italy. Let us know if you follow this  path since we want to  channel your financial contribution into our labs to fund the development of Chrono.
- If you can't make a donation but still like Chrono, you can still contribute by letting us know you used Chrono and sharing your experience with us
<br><br>

#### Can Chrono be compiled on platform X with compiler Y?
Currently we build Chrono under Windows&copy (32 and 64 bit with the MingW GNU&copy compiler,  Microsoft VisualC++&copy, and Intel&copy compilers) and Linux&copy (32 and 64 bit, GNU and Intel&copy compilers).
<br><br>

#### How is Chrono organized?
Chrono is designed with Chrono::Engine at its core. Beyond that, it has several [modules](@ref introduction_chrono) that augment and/or replace core Chrono::Engine functionality. For instance, Chrono::Vehicle adds support for vehicle modeling and simulation; Chrono::Parallel adds support for parallel computing when handling large systems, etc.
<br><br>

#### What is the difference between Chrono::Engine and Chrono::Parallel?
Chrono::Engine is the core of the Chrono middleware. Chrono::Engine was designed to primarily support rigid multibody dynamics, and in doing so to support as broad a spectrum of problems as possible. Chrono::Parallel is a module that can solve fast a subset of problems that Chrono::Engine can solve but can do so fast. In other words, it trades generality for speed.
<br><br>

#### What is the difference between Chrono units and Chrono modules?
There is no difference, we use these two terms interchangeably. 
<br><br>

#### What is a typical Chrono work flow?
The Chrono infrastructure is a modular set of libraries for multi-physics modeling and simulation. Some of these modules allow interfacing Chrono to external software, e.g., for pre-processing and post-processing data, displaying real-time simulations with OpenGL, or parsing Python commands. The power of Chrono stems from its open source attribute, which allows the user to mix-and-match modules to establish a work flow that matches his/her needs.
<br><br>

One possible work flow in Chrono: a model is defined in SolidWorks&copy; 
and enhanced with textures from some graphics package. 
The analysis might require a co-simulation with MATLAB&copy;. 
The results can be post-processed with Pov-Ray&copy; to obtained high quality movies.

![](workflow.png)
