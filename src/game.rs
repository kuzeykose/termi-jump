use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Player {
    pub x: f32,
    pub y: f32,
    pub vy: f32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Platform {
    pub x: u16,
    pub y: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GameState {
    pub player: Player,
    pub platforms: Vec<Platform>,
    pub score: u32,
    pub game_over: bool,
    pub last_landed_platform: Option<Platform>,
    pub camera_y: f32,
    pub target_camera_y: f32,
} 