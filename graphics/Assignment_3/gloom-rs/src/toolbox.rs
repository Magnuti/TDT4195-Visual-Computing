extern crate nalgebra_glm as glm;

pub struct Heading {
    pub x: f32,
    pub z: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
}

pub fn simple_heading_animation(time: f32) -> Heading {
    let t = time as f64;
    let step = 0.05f64;
    let path_size = 15f64;
    let circuit_speed = 0.8f64;

    let xpos = path_size*(2.0*t*circuit_speed).sin();
    let nextxpos = path_size*(2.0*(t+step)*circuit_speed).sin();
    let zpos = 3.0*path_size*(t*circuit_speed).cos();
    let nextzpos = 3.0*path_size*((t+step)*circuit_speed).cos();

    let delta_pos = glm::vec2(nextxpos-xpos, nextzpos-zpos);

    let yaw = std::f64::consts::PI + delta_pos.x.atan2(delta_pos.y);
    let pitch = -0.175 * glm::length(&delta_pos);
    let roll = (t*circuit_speed).cos() * 0.5;

    Heading {
        x: xpos as f32,
        z: zpos as f32,
        yaw: yaw as f32,
        pitch: pitch as f32,
        roll: roll as f32,
    }
}


