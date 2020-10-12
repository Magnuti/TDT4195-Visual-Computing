extern crate nalgebra_glm as glm;

use std::mem::ManuallyDrop;
use std::pin::Pin;

// Used to crete an unholy abomination upon which you should not cast your gaze.
// This ended up being a necessity due to wanting to keep the code written by students as "straight forward" as possible
// It is very very double plus ungood Rust, and intentionally leaks memory like a sieve. But it works, and you're more than welcome to pretend it doesn't exist!
// In case you're curious about how it works; It allocates memory on the heap (Box), promises to prevent it from being moved or deallocated until dropped (Pin) 
// and finally prevents the compiler from dropping it automatically at all (ManuallyDrop). If that sounds like a janky solution, it's because it is.
// Prettier, Rustier and better solutions were tried numerous times, but were all found wanting of having what I arbitrarily decided to be the required level of
// simplicity of use.
type Node = ManuallyDrop<Pin<Box<SceneNode>>>;

#[derive(Clone)]
pub struct SceneNode {
    pub position: glm::Vec3,
    pub rotation: glm::Vec3,
    pub scale: glm::Vec3,
    pub reference_point: glm::Vec3,

    pub current_transformation_matrix: glm::Mat4,

    pub vao_id: u32,
    pub index_count: i32,

    pub children: Vec<*mut SceneNode>,
}

impl SceneNode {
    pub fn new() -> Node {
        ManuallyDrop::new(Pin::new(Box::new(SceneNode {
            position: glm::zero(),
            rotation: glm::zero(),
            scale: glm::vec3(1.0, 1.0, 1.0),
            reference_point: glm::zero(),
            current_transformation_matrix: glm::identity(),
            vao_id: 0,
            index_count: -1,
            children: vec![],
        })))
    }
    pub fn from_vao(vao_id: u32, index_count: i32) -> Node {
        ManuallyDrop::new(Pin::new(Box::new(SceneNode {
            position: glm::zero(),
            rotation: glm::zero(),
            scale: glm::vec3(1.0, 1.0, 1.0),
            reference_point: glm::zero(),
            current_transformation_matrix: glm::identity(),
            vao_id, index_count,
            children: vec![],
        })))
    }
    pub fn add_child(&mut self, child: &SceneNode) {
        self.children.push(child as *const SceneNode as *mut SceneNode)
    }
    pub fn print(&self) {
        let m = self.current_transformation_matrix;
        let matrix_string = format!(
"
      {:.2}  {:.2}  {:.2}  {:.2}
      {:.2}  {:.2}  {:.2}  {:.2}
      {:.2}  {:.2}  {:.2}  {:.2}
      {:.2}  {:.2}  {:.2}  {:.2}
",
            m[0],m[4],m[8],m[12],
            m[1],m[5],m[9],m[13],
            m[2],m[6],m[10],m[14],
            m[3],m[7],m[11],m[15],
        );
        println!(
"SceneNode {{
    VAO:       {}
    Indices:   {}
    Children:  {}
    Position:  [{:.2}, {:.2}, {:.2}]
    Rotation:  [{:.2}, {:.2}, {:.2}]
    Reference: [{:.2}, {:.2}, {:.2}]
    Current Transformation Matrix: {}
}}",
            self.vao_id,
            self.index_count,
            self.children.len(),
            self.position.x,
            self.position.y,
            self.position.z,
            self.rotation.x,
            self.rotation.y,
            self.rotation.z,
            self.reference_point.x,
            self.reference_point.y,
            self.reference_point.z,
            matrix_string,
        );
    }
}
