// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Hammad Mazhar, Milad Rakhsha
// =============================================================================
// OpenGL window singleton, this class manages the opengl context and window
// =============================================================================

#ifndef CHOPENGLWINDOW_H
#define CHOPENGLWINDOW_H

#include "chrono_opengl/core/ChApiOpenGL.h"
#include "chrono_opengl/ChOpenGLViewer.h"

namespace chrono {

/// Namespace with classes for the Chrono::OpenGL module.
namespace opengl {

/// @addtogroup opengl_module
/// @{

/// Manager for the OpenGL context and window.
class CH_OPENGL_API ChOpenGLWindow {
  public:
    /// @brief Get the unique instance for the OpenGL window
    /// Call this function to get a pointer to the window
    static ChOpenGLWindow& getInstance() {
        static ChOpenGLWindow instance;
        return instance;
    }

    /// Initialize the window and set up the opengl viewer class
    void Initialize(int size_x,         ///< Width of window in pixels
                    int size_y,         ///< Height of window in pixels
                    const char* title,  ///< Window title string
                    ChSystem* msystem   ///< The ChSystem that is attached to this window
                    );

#ifdef CHRONO_FSI
    /// Initialize the window and set up the opengl viewer class if the FSI module is available
    void Initialize(int size_x,         ///< Width of window in pixels
                    int size_y,         ///< Height of window in pixels
                    const char* title,  ///< Window title string
                    ChSystem* msystem,  ///< The ChSystem that is attached to this window
                    chrono::fsi::ChSystemFsi* fsi_sys ///< The ChSystemFsi that is attached to this window
                    );
#endif

    /// This starts the drawing loop and takes control away from the main program
    /// This function is the easiest way to start rendering
    void StartDrawLoop(double time_step  // integration step size
                       );

    /// Perform a dynamics step, the user needs to use this so that pausing the
    /// simulation works correctly
    bool DoStepDynamics(double time_step  // integration step size
                        );

    /// Render the ChSystem and the HUD
    void Render();

    /// Check if the glfw context is still valid and the window has not been closed
    bool Active();

    /// Check if the simulation is running or paused
    bool Running();

    /// Pause simulation
    void Pause();

    // Set the camera position, look at and up vectors
    void SetCamera(ChVector<> pos,   // The position of the camera
                   ChVector<> look,  // The point that the camera is looking at
                   ChVector<> up,    // The up vector associated with the camera
                   float scale = 0.5f,
                   float near_clip_dist = 0.1f,
                   float far_clip_dist = 1000.0f) {
        viewer->render_camera.camera_position = glm::vec3(pos.x(), pos.y(), pos.z());
        viewer->render_camera.camera_look_at = glm::vec3(look.x(), look.y(), look.z());
        viewer->render_camera.camera_up = glm::vec3(up.x(), up.y(), up.z());
        viewer->render_camera.camera_scale = scale;
        viewer->render_camera.SetClipping(near_clip_dist, far_clip_dist);
    }

    void SetRenderMode(RenderMode mode) { viewer->render_mode = mode; }

    /// Provides the version of the opengl context along with driver information
    static void GLFWGetVersion(GLFWwindow* main_window  ///< A pointer to the window/context
                               );

    /// Pointer to the opengl viewer that handles rendering, text and user interaction
    ChOpenGLViewer* viewer;

  private:
    // Singleton constructor should be private so that a user cannot call it
    ChOpenGLWindow() {}
    // Singleton destructor should be private so that a user cannot call it
    ~ChOpenGLWindow() {}

    ChOpenGLWindow(ChOpenGLWindow const&);  // Don't Implement.
    void operator=(ChOpenGLWindow const&);  // Don't implement

    // GLFW error callback, returns error string
    static void CallbackError(int error, const char* description);

    // GLFW close callback, called when window is closed
    static void CallbackClose(GLFWwindow* window);

    // GLFW reshape callback, handles window resizing events
    static void CallbackReshape(GLFWwindow* window, int w, int h);

    // GLFW keyboard callback, handles keyboard events
    static void CallbackKeyboard(GLFWwindow* window, int key, int scancode, int action, int mode);

    // GLFW mouse button callback, handles mouse button events
    static void CallbackMouseButton(GLFWwindow* window, int button, int action, int mods);

    // GLFW mouse position callback, handles events generated by changes in
    // mouse position
    static void CallbackMousePos(GLFWwindow* window, double x, double y);

    // Pointer to the opengl context
    GLFWwindow* window;
    int poll_frame;
};

/// @} opengl_module
}
}

#endif
