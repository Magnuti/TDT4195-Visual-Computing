extern crate nalgebra_glm as glm;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::{mem, os::raw::c_void, ptr};

mod shader;
mod util;

use glutin::event::{
    DeviceEvent,
    ElementState::{Pressed, Released},
    Event, KeyboardInput,
    VirtualKeyCode::{self, *},
    WindowEvent,
};
use glutin::event_loop::ControlFlow;

const SCREEN_W: u32 = 800;
const SCREEN_H: u32 = 600;

// == // Helper functions to make interacting with OpenGL a little bit prettier. You *WILL* need these! // == //
// The names should be pretty self explanatory
fn byte_size_of_array<T>(val: &[T]) -> isize {
    std::mem::size_of_val(&val[..]) as isize
}

// Get the OpenGL-compatible pointer to an arbitrary array of numbers
fn pointer_to_array<T>(val: &[T]) -> *const c_void {
    &val[0] as *const T as *const c_void
}

// Get the size of the given type in bytes
fn size_of<T>() -> i32 {
    mem::size_of::<T>() as i32
}

// Get an offset in bytes for n units of type T
fn offset<T>(n: u32) -> *const c_void {
    (n * mem::size_of::<T>() as u32) as *const T as *const c_void
}

// Get a null pointer (equivalent to an offset of 0)
// ptr::null()

// == // Modify and complete the function below for the first task
unsafe fn create_vao(vertices: &Vec<f32>, indices: &Vec<u32>, colors: &Vec<f32>) -> u32 {
    /* Vertex array object */
    let mut vao_id = 0;
    gl::GenVertexArrays(1, &mut vao_id);
    gl::BindVertexArray(vao_id);

    /* Vertex buffer object for positions */
    let mut buffer_id = 0;
    gl::GenBuffers(1, &mut buffer_id);
    gl::BindBuffer(gl::ARRAY_BUFFER, buffer_id);
    // Transfer the data to the GPU i.e. fill the buffer
    gl::BufferData(
        gl::ARRAY_BUFFER,
        byte_size_of_array(vertices),
        pointer_to_array(vertices),
        gl::STATIC_DRAW,
    );
    let vertices_index = 0;
    gl::EnableVertexAttribArray(vertices_index);
    gl::VertexAttribPointer(
        vertices_index,
        3, // 3 coordinates -> [x, y, z]
        gl::FLOAT,
        gl::FALSE,   // Whether OpenGL should normalize the values in the buffer
        0, // All floats, so OpenGL fixes this. Specify a value != 0 if there are multiple types (e.g. float, integers) in one entry
        ptr::null(), // Array buffer offset
    );

    /* Vertex Buffer Object for colors */
    let mut buffer_color_id = 0;
    gl::GenBuffers(1, &mut buffer_color_id);
    gl::BindBuffer(gl::ARRAY_BUFFER, buffer_color_id);
    gl::BufferData(
        gl::ARRAY_BUFFER,
        byte_size_of_array(colors),
        pointer_to_array(colors),
        gl::STATIC_DRAW,
    );
    let colors_index = 1;
    gl::EnableVertexAttribArray(colors_index);
    // The VertexAttribPointer give the Vertex shader info about the data
    gl::VertexAttribPointer(
        colors_index,
        4, // 4 floats -> RGBA
        gl::FLOAT,
        gl::FALSE,   // Whether OpenGL should normalize the values in the buffer
        0, // All floats, so OpenGL fixes this. Specify a value != 0 if there are multiple types (e.g. float, integers) in one entry
        ptr::null(), // Array buffer offset
    );

    /* Index buffer */
    let mut index_buffer_id = 0;
    gl::GenBuffers(1, &mut index_buffer_id);
    gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, index_buffer_id);
    // Transfer the data to the GPU i.e. fill the buffer
    gl::BufferData(
        gl::ELEMENT_ARRAY_BUFFER,
        byte_size_of_array(indices),
        pointer_to_array(indices),
        gl::STATIC_DRAW,
    );

    // We do not need to call gl::VertexAttribPointer() to set up the index buffer

    return vao_id;
}

fn main() {
    // Set up the necessary objects to deal with windows and event handling
    let el = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new()
        .with_title("Gloom-rs")
        .with_resizable(false)
        .with_inner_size(glutin::dpi::LogicalSize::new(SCREEN_W, SCREEN_H));
    let cb = glutin::ContextBuilder::new().with_vsync(true);
    let windowed_context = cb.build_windowed(wb, &el).unwrap();
    // Uncomment these if you want to use the mouse for controls, but want it to be confined to the screen and/or invisible.
    // windowed_context.window().set_cursor_grab(true).expect("failed to grab cursor");
    // windowed_context.window().set_cursor_visible(false);
    // Set up a shared vector for keeping track of currently pressed keys
    let arc_pressed_keys = Arc::new(Mutex::new(Vec::<VirtualKeyCode>::with_capacity(10)));
    // Make a reference of this vector to send to the render thread
    let pressed_keys = Arc::clone(&arc_pressed_keys);

    // Set up shared tuple for tracking mouse movement between frames
    let arc_mouse_delta = Arc::new(Mutex::new((0f32, 0f32)));
    // Make a reference of this tuple to send to the render thread
    let mouse_delta = Arc::clone(&arc_mouse_delta);

    // Spawn a separate thread for rendering, so event handling doesn't block rendering
    let render_thread = thread::spawn(move || {
        // Acquire the OpenGL Context and load the function pointers. This has to be done inside of the rendering thread, because
        // an active OpenGL context cannot safely traverse a thread boundary
        let context = unsafe {
            let c = windowed_context.make_current().unwrap();
            gl::load_with(|symbol| c.get_proc_address(symbol) as *const _);
            c
        };

        // Set up openGL
        unsafe {
            gl::Enable(gl::DEPTH_TEST);
            gl::DepthFunc(gl::LESS);
            gl::Enable(gl::CULL_FACE);
            gl::Disable(gl::MULTISAMPLE);
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
            gl::Enable(gl::DEBUG_OUTPUT_SYNCHRONOUS);
            gl::DebugMessageCallback(Some(util::debug_callback), ptr::null());

            // Print some diagnostics
            println!(
                "{}: {}",
                util::get_gl_string(gl::VENDOR),
                util::get_gl_string(gl::RENDERER)
            );
            println!("OpenGL\t: {}", util::get_gl_string(gl::VERSION));
            println!(
                "GLSL\t: {}",
                util::get_gl_string(gl::SHADING_LANGUAGE_VERSION)
            );
        }

        // Task 1 data
        let vertices: Vec<f32> = vec![
            0.0, 0.0, 0.0, //       Center 0
            -0.5, 0.0, 0.0, //      Left center 1
            0.5, 0.0, 0.0, //       Right center 2
            -0.75, 0.5, 0.0, //     Top most left 3
            -0.25, 0.5, 0.0, //     Top left 4
            0.25, 0.5, 0.0, //      Top right 5
            0.75, 0.5, 0.0, //      Top most right 6
            -0.75, -0.5, 0.0, //    Bottom most left 7
            -0.25, -0.5, 0.0, //    Bottom left 8
            0.25, -0.5, 0.0, //     Bottom right 9
            0.75, -0.5, 0.0, //     Bottom most right 10
        ];

        // Note that we need to specify the coordinates in a non-clockwise order for triangles
        // https://www.khronos.org/opengl/wiki/Face_Culling
        let indices: Vec<u32> = vec![
            3, 1, 4, // Top left triangle
            4, 0, 5, // Top center triangle
            5, 2, 6, // Top right triangle
            7, 8, 1, // Bottom left triangle
            8, 9, 0, // Bottom center triangle
            9, 10, 2, // Bottom right triangle
        ];

        // One RGBA value per vertex in vertices i.e. 4 values here per 3 values in vertices
        let colors: Vec<f32> = vec![
            1.0, 1.0, 1.0, 1.0, // White
            0.0, 1.0, 1.0, 1.0, // GB
            1.0, 0.0, 1.0, 1.0, // RB
            1.0, 1.0, 0.0, 1.0, // RG
            0.0, 0.0, 1.0, 1.0, // B
            0.0, 1.0, 0.0, 1.0, // G
            1.0, 0.0, 0.0, 1.0, // R
            1.0, 1.0, 1.0, 0.5, // White 50%
            1.0, 0.0, 0.0, 0.5, // Red 50%
            0.0, 1.0, 0.0, 0.5, // Green 50%
            0.0, 0.0, 1.0, 0.5, // Blue 50%
        ];

        // Task 2 data
        /*let vertices: Vec<f32> = vec![
            0.6, -0.8, -1.2, //
            0.0, 0.4, 0.0, //
            -0.8, -0.2, 1.2, //
        ];
        let indices: Vec<u32> = vec![0, 1, 2];*/

        // Task 2d indices
        /*let indices: Vec<u32> = vec![
            3, 1, 4, // Top left triangle
            5, 2, 6, // Top right triangle
            8, 9, 0, // Bottom center triangle
        ];*/

        // == // Set up your VAO here
        let vao_id = unsafe { create_vao(&vertices, &indices, &colors) };

        // Basic usage of shader helper
        // The code below returns a shader object, which contains the field .program_id
        // The snippet is not enough to do the assignment, and will need to be modified (outside of just using the correct path), but it only needs to be called once
        // shader::ShaderBuilder::new().attach_file("./path/to/shader").link();
        unsafe {
            shader::ShaderBuilder::new()
                .attach_file("./shaders/simple.vert")
                .attach_file("./shaders/simple.frag")
                .link()
                .activate();
        }

        // Used to demonstrate keyboard handling -- feel free to remove
        let mut _arbitrary_number = 0.0;

        let first_frame_time = std::time::Instant::now();
        let mut last_frame_time = first_frame_time;
        // The main rendering loop
        loop {
            let now = std::time::Instant::now();
            let elapsed = now.duration_since(first_frame_time).as_secs_f32();
            let delta_time = now.duration_since(last_frame_time).as_secs_f32();
            last_frame_time = now;

            // Handle keyboard input
            if let Ok(keys) = pressed_keys.lock() {
                for key in keys.iter() {
                    match key {
                        VirtualKeyCode::A => {
                            _arbitrary_number += delta_time;
                        }
                        VirtualKeyCode::D => {
                            _arbitrary_number -= delta_time;
                        }

                        _ => {}
                    }
                }
            }
            // Handle mouse movement. delta contains the x and y movement of the mouse since last frame in pixels
            if let Ok(mut delta) = mouse_delta.lock() {
                *delta = (0.0, 0.0);
            }

            unsafe {
                gl::ClearColor(0.163, 0.163, 0.163, 1.0);
                gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

                // Issue the necessary commands to draw your scene here
                gl::BindVertexArray(vao_id);
                gl::DrawElements(
                    gl::TRIANGLES,
                    indices.len() as i32,
                    gl::UNSIGNED_INT,
                    ptr::null(),
                );
            }

            context.swap_buffers().unwrap();
        }
    });

    // Keep track of the health of the rendering thread
    let render_thread_healthy = Arc::new(RwLock::new(true));
    let render_thread_watchdog = Arc::clone(&render_thread_healthy);
    thread::spawn(move || {
        if !render_thread.join().is_ok() {
            if let Ok(mut health) = render_thread_watchdog.write() {
                println!("Render thread panicked!");
                *health = false;
            }
        }
    });

    // Start the event loop -- This is where window events get handled
    el.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        // Terminate program if render thread panics
        if let Ok(health) = render_thread_healthy.read() {
            if *health == false {
                *control_flow = ControlFlow::Exit;
            }
        }

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            // Keep track of currently pressed keys to send to the rendering thread
            Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: key_state,
                                virtual_keycode: Some(keycode),
                                ..
                            },
                        ..
                    },
                ..
            } => {
                if let Ok(mut keys) = arc_pressed_keys.lock() {
                    match key_state {
                        Released => {
                            if keys.contains(&keycode) {
                                let i = keys.iter().position(|&k| k == keycode).unwrap();
                                keys.remove(i);
                            }
                        }
                        Pressed => {
                            if !keys.contains(&keycode) {
                                keys.push(keycode);
                            }
                        }
                    }
                }

                // Handle escape separately
                match keycode {
                    Escape => {
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => {}
                }
            }
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                // Accumulate mouse movement
                if let Ok(mut position) = arc_mouse_delta.lock() {
                    *position = (position.0 + delta.0 as f32, position.1 + delta.1 as f32);
                }
            }
            _ => {}
        }
    });
}
