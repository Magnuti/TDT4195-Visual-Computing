extern crate nalgebra_glm as glm;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::{mem, os::raw::c_void, ptr};

mod mesh;
mod scene_graph;
mod shader;
mod toolbox;
mod util;

use glutin::event::{
    DeviceEvent,
    ElementState::{Pressed, Released},
    Event, KeyboardInput,
    VirtualKeyCode::{self, *},
    WindowEvent,
};
use glutin::event_loop::ControlFlow;

use std::mem::ManuallyDrop;
use std::pin::Pin;

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
unsafe fn create_vao(
    vertices: &Vec<f32>,
    indices: &Vec<u32>,
    colors: &Vec<f32>,
    normals: &Vec<f32>,
) -> u32 {
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

    /* Vertex Buffer Object for normals */
    let mut buffer_normal_id = 0;
    gl::GenBuffers(1, &mut buffer_normal_id);
    gl::BindBuffer(gl::ARRAY_BUFFER, buffer_normal_id);
    gl::BufferData(
        gl::ARRAY_BUFFER,
        byte_size_of_array(normals),
        pointer_to_array(normals),
        gl::STATIC_DRAW,
    );
    let normals_index = 2;
    gl::EnableVertexAttribArray(normals_index);
    // The VertexAttribPointer give the Vertex shader info about the data
    gl::VertexAttribPointer(
        normals_index,
        3, // 3 floats -> XYZ
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

unsafe fn create_vao_from_mesh(mesh: &mesh::Mesh) -> u32 {
    return create_vao(&mesh.vertices, &mesh.indices, &mesh.colors, &mesh.normals);
}

unsafe fn draw_scene(root: &scene_graph::SceneNode, view_projection_matrix: &glm::Mat4) {
    // Check if node is drawable, set uniforms, draw
    // Checks if there are elements in the indicies array
    if root.index_count > 0 {
        gl::BindVertexArray(root.vao_id);

        let mvp_matrix = view_projection_matrix * root.current_transformation_matrix;

        gl::UniformMatrix4fv(3, 1, gl::FALSE, mvp_matrix.as_ptr()); // Pass 1 matrix to layout (location = 3) - MVP matrix
        gl::UniformMatrix4fv(4, 1, gl::FALSE, root.current_transformation_matrix.as_ptr()); // Pass 1 matrix to layout (location = 4) - model matrix

        gl::DrawElements(
            gl::TRIANGLES,
            root.index_count,
            gl::UNSIGNED_INT,
            ptr::null(),
        );
    }

    // Recurse
    for &child in &root.children {
        draw_scene(&*child, view_projection_matrix);
    }
}

unsafe fn update_node_transformations(
    root: &mut scene_graph::SceneNode,
    transformation_so_far: &glm::Mat4,
) {
    // ! Do not submit this
    // Construct the correct transformation matrix
    // root.current_transformation_matrix = root.current_transformation_matrix * transformation_so_far;
    root.current_transformation_matrix.fill_with_identity();
    root.current_transformation_matrix =
        glm::scale(&root.current_transformation_matrix, &root.scale);
    root.current_transformation_matrix =
        glm::translate(&root.current_transformation_matrix, &root.position);
    root.current_transformation_matrix =
        glm::translate(&root.current_transformation_matrix, &&root.reference_point);
    root.current_transformation_matrix =
        glm::rotate_x(&root.current_transformation_matrix, root.rotation.x);
    root.current_transformation_matrix =
        glm::rotate_y(&root.current_transformation_matrix, root.rotation.y);
    root.current_transformation_matrix =
        glm::rotate_z(&root.current_transformation_matrix, root.rotation.z);
    root.current_transformation_matrix =
        glm::translate(&root.current_transformation_matrix, &-&root.reference_point);
    root.current_transformation_matrix =
        transformation_so_far * &root.current_transformation_matrix;

    // Update the node's transformation matrix
    // Recurse
    for &child in &root.children {
        update_node_transformations(&mut *child, &root.current_transformation_matrix);
    }
}

struct Helicopter {
    body: mem::ManuallyDrop<std::pin::Pin<std::boxed::Box<scene_graph::SceneNode>>>,
    // door: mem::ManuallyDrop<std::pin::Pin<std::boxed::Box<scene_graph::SceneNode>>>,
    main_rotor: mem::ManuallyDrop<std::pin::Pin<std::boxed::Box<scene_graph::SceneNode>>>,
    tail_rotor: mem::ManuallyDrop<std::pin::Pin<std::boxed::Box<scene_graph::SceneNode>>>,
}

fn create_helicopter(
    helicopter: &mesh::Helicopter,
    terrain: &mut scene_graph::SceneNode,
) -> Helicopter {
    let helicopter_body_vao_id = unsafe { create_vao_from_mesh(&helicopter.body) };
    let helicopter_door_vao_id = unsafe { create_vao_from_mesh(&helicopter.door) };
    let helicopter_main_rotor_vao_id = unsafe { create_vao_from_mesh(&helicopter.main_rotor) };
    let helicopter_tail_rotor_vao_id = unsafe { create_vao_from_mesh(&helicopter.tail_rotor) };

    // Setup helicopter
    let mut helicopter_body_scene_node =
        scene_graph::SceneNode::from_vao(helicopter_body_vao_id, helicopter.body.index_count);
    let helicopter_door_scene_node =
        scene_graph::SceneNode::from_vao(helicopter_door_vao_id, helicopter.door.index_count);
    let helicopter_main_rotor_scene_node = scene_graph::SceneNode::from_vao(
        helicopter_main_rotor_vao_id,
        helicopter.main_rotor.index_count,
    );
    let mut helicopter_tail_rotor_scene_node = scene_graph::SceneNode::from_vao(
        helicopter_tail_rotor_vao_id,
        helicopter.tail_rotor.index_count,
    );
    helicopter_tail_rotor_scene_node.reference_point = glm::Vec3::new(0.35, 2.3, 10.4);

    terrain.add_child(&helicopter_body_scene_node);
    helicopter_body_scene_node.add_child(&helicopter_door_scene_node);
    helicopter_body_scene_node.add_child(&helicopter_main_rotor_scene_node);
    helicopter_body_scene_node.add_child(&helicopter_tail_rotor_scene_node);

    return Helicopter {
        body: helicopter_body_scene_node,
        // door: helicopter_door_scene_node,
        main_rotor: helicopter_main_rotor_scene_node,
        tail_rotor: helicopter_tail_rotor_scene_node,
    };
}

fn apply_motion(heli: &mut Helicopter, elapsed_time: f32, offset: f32) {
    rotate_main_rotor(&mut heli.main_rotor, elapsed_time);
    rotate_tail_rotor(&mut heli.tail_rotor, elapsed_time);
    change_heading(
        &mut heli.body,
        toolbox::simple_heading_animation(elapsed_time + offset),
    );
}

fn rotate_main_rotor(node: &mut scene_graph::SceneNode, elapsed_time: f32) {
    node.rotation = glm::vec3(0.0, 15.0 * elapsed_time, 0.0);
}

fn rotate_tail_rotor(node: &mut scene_graph::SceneNode, elapsed_time: f32) {
    node.rotation = glm::vec3(20.0 * elapsed_time, 0.0, 0.0);
}

fn change_heading(node: &mut scene_graph::SceneNode, heading: toolbox::Heading) {
    node.position.x = heading.x;
    node.position.z = heading.z;
    node.rotation = glm::vec3(heading.pitch, heading.yaw, heading.roll);
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

        let perspective: glm::Mat4 = glm::perspective(
            (SCREEN_W as f32) / (SCREEN_H as f32), // Aspect ratio = width/height
            (60.0 * 3.14) / 180.0,                 // 60 degress FOV, but the function uses radians
            1.0,                                   //
            1000.0,                                //
        );

        /* Assignment 3 */
        let lunar_surface: mesh::Mesh = mesh::Terrain::load("./resources/lunarsurface.obj");

        let lunar_surface_vao_id = unsafe { create_vao_from_mesh(&lunar_surface) };

        let helicopter: mesh::Helicopter = mesh::Helicopter::load("./resources/helicopter.obj");

        /* Scene graph start */
        // Setup terrain and root
        let mut root_scene_node = scene_graph::SceneNode::new();
        let mut terrain_scene_node =
            scene_graph::SceneNode::from_vao(lunar_surface_vao_id, lunar_surface.index_count);
        root_scene_node.add_child(&terrain_scene_node);

        let mut heli_1 = create_helicopter(&helicopter, &mut terrain_scene_node);
        let mut heli_2 = create_helicopter(&helicopter, &mut terrain_scene_node);
        let mut heli_3 = create_helicopter(&helicopter, &mut terrain_scene_node);
        let mut heli_4 = create_helicopter(&helicopter, &mut terrain_scene_node);
        let mut heli_5 = create_helicopter(&helicopter, &mut terrain_scene_node);

        /* Scene graph end */

        /* End assignment 3 */

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

        // World coordinates according to the camera
        let mut x = 0.0;
        let mut y = 0.0;
        let mut z = 0.0;
        let mut yaw = 0.0; // Left-right rotation (parallell to the floor)
        let mut pitch = 0.0; // Up-down rotation
        let mut roll = 0.0; // Roll (left-right roll like in a boat)

        let first_frame_time = std::time::Instant::now();
        let mut last_frame_time = first_frame_time;
        // The main rendering loop
        loop {
            let now = std::time::Instant::now();
            let elapsed = now.duration_since(first_frame_time).as_secs_f32();
            let delta_time = now.duration_since(last_frame_time).as_secs_f32();
            last_frame_time = now;

            let z_speed = 20.0;
            let x_speed = 20.0;
            let y_speed = 20.0;
            let pitch_speed = 2.0;
            let yaw_speed = 1.5;
            let roll_speed = 1.0;

            // Handle keyboard input
            if let Ok(keys) = pressed_keys.lock() {
                for key in keys.iter() {
                    match key {
                        VirtualKeyCode::A => {
                            x += delta_time * x_speed;
                        }
                        VirtualKeyCode::D => {
                            x -= delta_time * x_speed;
                        }
                        VirtualKeyCode::W => {
                            z += delta_time * z_speed;
                        }
                        VirtualKeyCode::S => {
                            z -= delta_time * z_speed;
                        }
                        VirtualKeyCode::LShift => {
                            y -= delta_time * y_speed;
                        }
                        VirtualKeyCode::LControl => {
                            y += delta_time * y_speed;
                        }
                        VirtualKeyCode::Up => {
                            pitch += delta_time * pitch_speed;
                        }
                        VirtualKeyCode::Down => {
                            pitch -= delta_time * pitch_speed;
                        }
                        VirtualKeyCode::Right => {
                            yaw += delta_time * yaw_speed;
                        }
                        VirtualKeyCode::Left => {
                            yaw -= delta_time * yaw_speed;
                        }
                        VirtualKeyCode::E => {
                            roll += delta_time * roll_speed;
                        }
                        VirtualKeyCode::Q => {
                            roll -= delta_time * roll_speed;
                        }
                        _ => {}
                    }
                }
            }
            // Handle mouse movement. delta contains the x and y movement of the mouse since last frame in pixels
            if let Ok(mut delta) = mouse_delta.lock() {
                *delta = (0.0, 0.0);
            }

            let mut view_projection_matrix: glm::Mat4 = perspective;

            // Perform the camera transformation before rendering
            view_projection_matrix = glm::rotate_y(&view_projection_matrix, yaw);
            view_projection_matrix = glm::rotate_x(&view_projection_matrix, pitch);
            view_projection_matrix = glm::rotate_z(&view_projection_matrix, roll);
            view_projection_matrix = glm::translate(&view_projection_matrix, &glm::vec3(x, y, z));

            // An offset of 0.8 seconds seems to be the sweetspot between not crashing in the turn and not crashing in the cross
            apply_motion(&mut heli_1, elapsed, 0.0);
            apply_motion(&mut heli_2, elapsed, 0.8);
            apply_motion(&mut heli_3, elapsed, 1.6);
            apply_motion(&mut heli_4, elapsed, 2.4);
            apply_motion(&mut heli_5, elapsed, 3.2);

            unsafe {
                update_node_transformations(&mut root_scene_node, &glm::identity());
            }

            unsafe {
                gl::ClearColor(0.163, 0.163, 0.163, 1.0);
                gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

                draw_scene(&root_scene_node, &view_projection_matrix);
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
