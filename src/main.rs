use crossterm::{
    cursor,
    event::{self, Event, KeyCode},
    execute,
    terminal::{self, ClearType},
};
use rand::Rng;
use std::io::{stdout, Write};
use std::time::{Duration, Instant};
use std::thread;

mod ai;
mod game;

use ai::{Experience, ReplayBuffer};
use game::{GameState, Player, Platform};

const WIDTH: u16 = 40;
const HEIGHT: u16 = 30;
const PLATFORM_COUNT: usize = 8;
const PLATFORM_WIDTH: u16 = 8;
const GRAVITY: f32 = 0.2;
const JUMP_STRENGTH: f32 = 2.0;
const HORIZONTAL_SPEED: f32 = 2.0;

fn main() -> std::io::Result<()> {
    let mut stdout = stdout();
    terminal::enable_raw_mode()?;
    execute!(
        stdout,
        terminal::EnterAlternateScreen,
        terminal::Clear(ClearType::All),
        cursor::Hide
    )?;

    let mut game_state = GameState {
        player: Player {
            x: (WIDTH / 2) as f32,
            y: (HEIGHT - 5) as f32,
            vy: 0.0,
        },
        platforms: generate_platforms(PLATFORM_COUNT),
        score: 0,
        game_over: false,
        last_landed_platform: None,
        camera_y: 0.0,
        target_camera_y: 0.0,
    };
    
    let agent = ai::Agent::new().expect("Error creating agent");
    
    let mut replay_buffer = ReplayBuffer::new(10000);
    let batch_size = 32;
    let discount_factor = 0.95;
    let mut frame_count = 0;

    let starting_platform = Platform {
        x: game_state.player.x as u16 - PLATFORM_WIDTH / 2,
        y: game_state.player.y + 1.0,
    };
    game_state.platforms[0] = starting_platform;
    game_state.last_landed_platform = Some(starting_platform);

    let tick_rate = Duration::from_millis(50);
    while !game_state.game_over {
        let start_time = Instant::now();
        let state_vec = agent.get_feature_vector(&game_state);

        // --- INPUT ---
        if event::poll(Duration::from_millis(0))? {
            if let Event::Key(key_event) = event::read()? {
                if key_event.code == KeyCode::Char('q') || key_event.code == KeyCode::Esc {
                    game_state.game_over = true;
                }
            }
        }
        
        let action = agent.decide(&game_state);
        let mut reward = 0.0;

        // --- UPDATE ---
        let last_y = game_state.player.y;

        if action == -1 {
            game_state.player.x -= HORIZONTAL_SPEED;
        } else if action == 1 {
            game_state.player.x += HORIZONTAL_SPEED;
        }

        game_state.player.vy += GRAVITY;
        game_state.player.y += game_state.player.vy;

        // Smooth camera movement
        let camera_diff = game_state.target_camera_y - game_state.camera_y;
        if camera_diff.abs() > 0.01 {
            game_state.camera_y += camera_diff * 0.1; // Smoothing factor
        } else {
            game_state.camera_y = game_state.target_camera_y;
        }

        if game_state.player.x < 1.0 {
            game_state.player.x = (WIDTH - 2) as f32;
        } else if game_state.player.x >= (WIDTH - 1) as f32 {
            game_state.player.x = 1.0;
        }
        
        if game_state.player.vy > 0.0 {
            for p in &game_state.platforms {
                let player_x = game_state.player.x as u16;
                let player_y = game_state.player.y;

                if player_x >= p.x && player_x < p.x + PLATFORM_WIDTH {
                    let previous_y = player_y - game_state.player.vy;
                    if previous_y <= p.y && player_y >= p.y {
                        game_state.player.vy = -JUMP_STRENGTH;
                        game_state.player.y = p.y;
                        
                        if game_state.last_landed_platform.map_or(true, |last_p| last_p.y > p.y) {
                            reward += 10.0; // Reward for landing on a new, higher platform
                        }
                        game_state.last_landed_platform = Some(*p);

                        // If we landed on a high platform, schedule a scroll.
                        let screen_y = p.y - game_state.camera_y;
                        if screen_y < (HEIGHT / 2) as f32 {
                            game_state.target_camera_y = p.y - (HEIGHT / 2) as f32;
                        }
                        break;
                    }
                }
            }
        }

        if game_state.player.y > game_state.camera_y + HEIGHT as f32 {
            game_state.game_over = true;
            reward = -100.0; // Penalty for dying
        }

        // Reward for vertical progress
        reward += (last_y - game_state.player.y) * 0.1;

        let next_state_vec = agent.get_feature_vector(&game_state);
        replay_buffer.push(Experience {
            state: state_vec,
            action: action,
            reward: reward,
            next_state: next_state_vec,
            done: game_state.game_over,
        });
        
        frame_count += 1;
        if frame_count % 10 == 0 { // Train every 10 frames
            if let Some(batch) = replay_buffer.sample(batch_size) {
                if let Err(e) = agent.train(&batch, discount_factor) {
                    eprintln!("Error training agent: {}", e);
                }
            }
        }

        // --- Platform Management ---
        let bottom_of_view = game_state.camera_y + HEIGHT as f32;
        game_state.platforms.retain(|p| p.y < bottom_of_view + 40.0);

        let upcoming_platform_count = game_state.platforms.iter().filter(|p| p.y < game_state.camera_y).count();

        if upcoming_platform_count < PLATFORM_COUNT / 2 {
             let mut new_platforms = generate_new_platforms(PLATFORM_COUNT * 2, &game_state.platforms);
             game_state.platforms.append(&mut new_platforms);
        }

        game_state.score = (-game_state.camera_y.min(0.0)) as u32;

        // --- DRAW ---
        execute!(stdout, terminal::Clear(ClearType::All))?;
        draw_boundaries(&mut stdout)?;
        draw_platforms(&mut stdout, &game_state.platforms, game_state.camera_y)?;
        draw_player(&mut stdout, &game_state.player, game_state.camera_y)?;
        draw_score(&mut stdout, game_state.score)?;
        stdout.flush()?;

        let elapsed = start_time.elapsed();
        if let Some(time_to_wait) = tick_rate.checked_sub(elapsed) {
            thread::sleep(time_to_wait);
        }
    }

    // --- GAME OVER ---
    execute!(
        stdout,
        cursor::MoveTo(WIDTH / 2 - 5, HEIGHT / 2),
        crossterm::style::Print("Game Over!"),
        cursor::MoveTo(WIDTH / 2 - 10, HEIGHT / 2 + 1),
        crossterm::style::Print(format!("Final Score: {}", game_state.score)),
        cursor::MoveTo(0, HEIGHT)
    )?;

    // Cleanup
    execute!(
        stdout,
        cursor::Show,
        terminal::LeaveAlternateScreen
    )?;
    terminal::disable_raw_mode()?;

    Ok(())
}

fn generate_platforms(count: usize) -> Vec<Platform> {
    let mut platforms = Vec::with_capacity(count);
    let mut rng = rand::thread_rng();
    for _ in 0..count {
        platforms.push(Platform {
            x: rng.gen_range(1..WIDTH - PLATFORM_WIDTH - 1),
            y: rng.gen_range(0..HEIGHT) as f32,
        });
    }
    platforms
}

fn generate_new_platforms(count: usize, existing_platforms: &[Platform]) -> Vec<Platform> {
    let mut platforms = Vec::with_capacity(count);
    let mut rng = rand::thread_rng();

    let mut path_anchor = existing_platforms.iter()
        .min_by(|a, b| a.y.partial_cmp(&b.y).unwrap())
        .map(|p| (p.x, p.y))
        .unwrap_or((WIDTH / 2, 0.0));

    let path_len = count / 2;
    if path_len == 0 { return vec![]; }

    for _ in 0..path_len {
        let vertical_dist = rng.gen_range(4.0..=9.0);
        let next_y = path_anchor.1 - vertical_dist;

        let max_horizontal_dist = 9;
        let x_min = path_anchor.0.saturating_sub(max_horizontal_dist);
        let x_max = (path_anchor.0 + max_horizontal_dist).min(WIDTH - PLATFORM_WIDTH - 1);
        let next_x = if x_min >= x_max { x_min } else { rng.gen_range(x_min..=x_max) };

        platforms.push(Platform { x: next_x, y: next_y });
        path_anchor = (next_x, next_y);
    }

    let highest_y_in_path = platforms.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);
    let lowest_y_in_path = platforms.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);

    for _ in path_len..count {
        platforms.push(Platform {
            x: rng.gen_range(1..WIDTH - PLATFORM_WIDTH - 1),
            y: rng.gen_range(lowest_y_in_path..=highest_y_in_path),
        });
    }

    platforms
}

fn draw_boundaries(stdout: &mut impl Write) -> std::io::Result<()> {
    for y in 0..HEIGHT {
        execute!(
            stdout,
            cursor::MoveTo(0, y),
            crossterm::style::Print("|"),
            cursor::MoveTo(WIDTH - 1, y),
            crossterm::style::Print("|")
        )?;
    }
    Ok(())
}

fn draw_player(stdout: &mut impl Write, player: &Player, camera_y: f32) -> std::io::Result<()> {
    let screen_y = (player.y - camera_y) as u16;
    if screen_y < HEIGHT {
        execute!(
            stdout,
            cursor::MoveTo(player.x as u16, screen_y),
            crossterm::style::Print("^")
        )?;
    }
    Ok(())
}

fn draw_platforms(stdout: &mut impl Write, platforms: &[Platform], camera_y: f32) -> std::io::Result<()> {
    println!("\n\n\n Drawing platforms: {:?}", platforms);
    for p in platforms {
        let screen_y = (p.y - camera_y) as u16;
        if screen_y < HEIGHT {
            execute!(
                stdout,
                cursor::MoveTo(p.x, screen_y),
                crossterm::style::Print("========")
            )?;
        }
    }
    Ok(())
}

fn draw_score(stdout: &mut impl Write, score: u32) -> std::io::Result<()> {
    execute!(
        stdout,
        cursor::MoveTo(0, 0),
        crossterm::style::Print(format!("Score: {}", score))
    )
}
