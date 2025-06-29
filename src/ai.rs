// This module will contain the AI agent that plays the game.

use crate::game::GameState;
use rand::Rng;
use tensorflow::{Graph, Session, SessionOptions, SessionRunArgs, Status, Tensor};

pub struct Agent {
    session: Session,
    graph: Graph,
    input_op: tensorflow::Operation,
    output_op: tensorflow::Operation,
    
    // Ops for training
    target_q_op: tensorflow::Operation,
    action_op: tensorflow::Operation,
    train_op: tensorflow::Operation,
}

#[derive(Clone)]
pub struct Experience {
    pub state: Vec<f32>,
    pub action: i32,
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub done: bool,
}

pub struct ReplayBuffer {
    buffer: std::collections::VecDeque<Experience>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        ReplayBuffer {
            buffer: std::collections::VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, experience: Experience) {
        if self.buffer.len() == self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }

    pub fn sample(&self, batch_size: usize) -> Option<Vec<Experience>> {
        if self.buffer.len() < batch_size {
            return None;
        }
        let mut rng = rand::thread_rng();
        let samples = rand::seq::index::sample(&mut rng, self.buffer.len(), batch_size);
        let batch = samples.iter().map(|i| self.buffer[i].clone()).collect();
        Some(batch)
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

impl Agent {
    pub fn new() -> Result<Self, Status> {
        let mut graph = Graph::new();
        
        let feature_count = 3 + 2 * 8; // player (3) + 8 platforms (2 each)
        
        let mut input_op_desc = graph.new_operation("Placeholder", "input")?;
        input_op_desc.set_attr_type("dtype", tensorflow::DataType::Float)?;
        input_op_desc.set_attr_shape("shape", &tensorflow::Shape::from(&[1, feature_count as i64][..]))?;
        let input = input_op_desc.finish()?;
        
        let hidden_nodes = 16;
        let mut rng = rand::thread_rng();

        // --- Variables ---
        let mut w1_op_desc = graph.new_operation("VariableV2", "weights1")?;
        w1_op_desc.set_attr_shape("shape", &tensorflow::Shape::from(&[feature_count as i64, hidden_nodes as i64][..]))?;
        w1_op_desc.set_attr_type("dtype", tensorflow::DataType::Float)?;
        let w1 = w1_op_desc.finish()?;

        let mut b1_op_desc = graph.new_operation("VariableV2", "biases1")?;
        b1_op_desc.set_attr_shape("shape", &tensorflow::Shape::from(&[hidden_nodes as i64][..]))?;
        b1_op_desc.set_attr_type("dtype", tensorflow::DataType::Float)?;
        let b1 = b1_op_desc.finish()?;
        
        let mut w2_op_desc = graph.new_operation("VariableV2", "weights2")?;
        w2_op_desc.set_attr_shape("shape", &tensorflow::Shape::from(&[hidden_nodes as i64, 3][..]))?;
        w2_op_desc.set_attr_type("dtype", tensorflow::DataType::Float)?;
        let w2 = w2_op_desc.finish()?;

        let mut b2_op_desc = graph.new_operation("VariableV2", "biases2")?;
        b2_op_desc.set_attr_shape("shape", &tensorflow::Shape::from(&[3i64][..]))?;
        b2_op_desc.set_attr_type("dtype", tensorflow::DataType::Float)?;
        let b2 = b2_op_desc.finish()?;

        // --- Initialization Ops ---
        let w1_init_values: Vec<f32> = (0..feature_count * hidden_nodes).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let mut w1_init_op_desc = graph.new_operation("Const", "w1_init")?;
        w1_init_op_desc.set_attr_tensor("value", Tensor::new(&[feature_count as u64, hidden_nodes as u64]).with_values(&w1_init_values)?)?;
        w1_init_op_desc.set_attr_type("dtype", tensorflow::DataType::Float)?;
        let w1_init = w1_init_op_desc.finish()?;
        
        let b1_init_values: Vec<f32> = (0..hidden_nodes).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let mut b1_init_op_desc = graph.new_operation("Const", "b1_init")?;
        b1_init_op_desc.set_attr_tensor("value", Tensor::new(&[hidden_nodes as u64]).with_values(&b1_init_values)?)?;
        b1_init_op_desc.set_attr_type("dtype", tensorflow::DataType::Float)?;
        let b1_init = b1_init_op_desc.finish()?;

        let w2_init_values: Vec<f32> = (0..hidden_nodes * 3).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let mut w2_init_op_desc = graph.new_operation("Const", "w2_init")?;
        w2_init_op_desc.set_attr_tensor("value", Tensor::new(&[hidden_nodes as u64, 3]).with_values(&w2_init_values)?)?;
        w2_init_op_desc.set_attr_type("dtype", tensorflow::DataType::Float)?;
        let w2_init = w2_init_op_desc.finish()?;
        
        let b2_init_values: Vec<f32> = (0..3).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let mut b2_init_op_desc = graph.new_operation("Const", "b2_init")?;
        b2_init_op_desc.set_attr_tensor("value", Tensor::new(&[3]).with_values(&b2_init_values)?)?;
        b2_init_op_desc.set_attr_type("dtype", tensorflow::DataType::Float)?;
        let b2_init = b2_init_op_desc.finish()?;

        let mut w1_assign_op_desc = graph.new_operation("Assign", "w1_assign")?;
        w1_assign_op_desc.add_input(w1.clone());
        w1_assign_op_desc.add_input(w1_init);
        let w1_assign = w1_assign_op_desc.finish()?;

        let mut b1_assign_op_desc = graph.new_operation("Assign", "b1_assign")?;
        b1_assign_op_desc.add_input(b1.clone());
        b1_assign_op_desc.add_input(b1_init);
        let b1_assign = b1_assign_op_desc.finish()?;

        let mut w2_assign_op_desc = graph.new_operation("Assign", "w2_assign")?;
        w2_assign_op_desc.add_input(w2.clone());
        w2_assign_op_desc.add_input(w2_init);
        let w2_assign = w2_assign_op_desc.finish()?;

        let mut b2_assign_op_desc = graph.new_operation("Assign", "b2_assign")?;
        b2_assign_op_desc.add_input(b2.clone());
        b2_assign_op_desc.add_input(b2_init);
        let b2_assign = b2_assign_op_desc.finish()?;

        // --- Model Graph ---
        let mut matmul1_op_desc = graph.new_operation("MatMul", "matmul1")?;
        matmul1_op_desc.add_input(input.clone());
        matmul1_op_desc.add_input(w1.clone());
        let matmul1 = matmul1_op_desc.finish()?;

        let mut add1_op_desc = graph.new_operation("Add", "add1")?;
        add1_op_desc.add_input(matmul1);
        add1_op_desc.add_input(b1.clone());
        let add1 = add1_op_desc.finish()?;
        
        let mut relu1_op_desc = graph.new_operation("Relu", "relu1")?;
        relu1_op_desc.add_input(add1);
        let relu1 = relu1_op_desc.finish()?;

        let mut matmul2_op_desc = graph.new_operation("MatMul", "matmul2")?;
        matmul2_op_desc.add_input(relu1);
        matmul2_op_desc.add_input(w2.clone());
        let matmul2 = matmul2_op_desc.finish()?;

        let mut output_op_desc = graph.new_operation("Add", "output")?;
        output_op_desc.add_input(matmul2);
        output_op_desc.add_input(b2.clone());
        let output = output_op_desc.finish()?;
        
        // --- Training Graph ---
        let mut target_q_op_desc = graph.new_operation("Placeholder", "target_q")?;
        target_q_op_desc.set_attr_type("dtype", tensorflow::DataType::Float)?;
        target_q_op_desc.set_attr_shape("shape", &tensorflow::Shape::from(&[-1i64][..]))?;
        let target_q = target_q_op_desc.finish()?;
            
        let mut action_op_desc = graph.new_operation("Placeholder", "action")?;
        action_op_desc.set_attr_type("dtype", tensorflow::DataType::Int32)?;
        action_op_desc.set_attr_shape("shape", &tensorflow::Shape::from(&[-1i64, 1i64][..]))?;
        let action = action_op_desc.finish()?;
        
        let mut responsible_output_op_desc = graph.new_operation("GatherNd", "responsible_output")?;
        responsible_output_op_desc.add_input(output.clone());
        responsible_output_op_desc.add_input(action.clone());
        let responsible_output = responsible_output_op_desc.finish()?;

        let mut loss_sub_op_desc = graph.new_operation("Sub", "loss_sub")?;
        loss_sub_op_desc.add_input(target_q.clone());
        loss_sub_op_desc.add_input(responsible_output);
        let loss_sub = loss_sub_op_desc.finish()?;

        let mut loss_sq_op_desc = graph.new_operation("Square", "loss_sq")?;
        loss_sq_op_desc.add_input(loss_sub);
        let loss_sq = loss_sq_op_desc.finish()?;

        let mut reduction_indices_op_desc = graph.new_operation("Const", "reduction_indices")?;
        reduction_indices_op_desc.set_attr_tensor("value", Tensor::<i32>::new(&[])
            .with_values(&[])?)?;
        reduction_indices_op_desc.set_attr_type("dtype", tensorflow::DataType::Int32)?;
        let reduction_indices = reduction_indices_op_desc.finish()?;

        let mut loss_op_desc = graph.new_operation("Mean", "loss")?;
        loss_op_desc.add_input(loss_sq);
        loss_op_desc.add_input(reduction_indices);
        let loss = loss_op_desc.finish()?;
        
        let learning_rate = 0.001f32;
        let grads = graph.add_gradients(None, &[loss.clone().into()], &[w1.clone().into(), b1.clone().into(), w2.clone().into(), b2.clone().into()], None)?;
        
        let mut lr_const_op = graph.new_operation("Const", "lr")?;
        lr_const_op.set_attr_tensor("value", Tensor::from(learning_rate))?;
        lr_const_op.set_attr_type("dtype", tensorflow::DataType::Float)?;
        let lr_op = lr_const_op.finish()?;

        let mut train_op_desc = graph.new_operation("ApplyGradientDescent", "train")?;
        train_op_desc.add_input(w1.clone());
        train_op_desc.add_input(lr_op.clone());
        train_op_desc.add_input(grads[0].clone().unwrap());

        train_op_desc.add_input(b1.clone());
        train_op_desc.add_input(lr_op.clone());
        train_op_desc.add_input(grads[1].clone().unwrap());
        
        train_op_desc.add_input(w2.clone());
        train_op_desc.add_input(lr_op.clone());
        train_op_desc.add_input(grads[2].clone().unwrap());
        
        train_op_desc.add_input(b2.clone());
        train_op_desc.add_input(lr_op.clone());
        train_op_desc.add_input(grads[3].clone().unwrap());
        
        let train_op = train_op_desc.finish()?;

        let mut init_op = graph.new_operation("NoOp", "init")?;
        init_op.add_control_input(&w1_assign);
        init_op.add_control_input(&b1_assign);
        init_op.add_control_input(&w2_assign);
        init_op.add_control_input(&b2_assign);
        let init_op = init_op.finish()?;
        
        let session = Session::new(&SessionOptions::new(), &graph)?;
        
        let mut args = SessionRunArgs::new();
        args.add_target(&init_op);
        session.run(&mut args)?;

        Ok(Agent {
            session,
            graph,
            input_op: input,
            output_op: output,
            target_q_op: target_q,
            action_op: action,
            train_op,
        })
    }

    pub fn train(&self, batch: &[Experience], discount_factor: f32) -> Result<(), Status> {
        let mut states_vec = Vec::new();
        let mut actions_vec = Vec::new();
        let mut target_qs_vec = Vec::new();

        for (i, experience) in batch.iter().enumerate() {
            states_vec.extend_from_slice(&experience.state);
            actions_vec.push([i as i32, experience.action as i32]);
            
            let next_state_tensor = Tensor::new(&[1, experience.next_state.len() as u64]).with_values(&experience.next_state)?;
            let mut next_q_args = SessionRunArgs::new();
            next_q_args.add_feed(&self.input_op, 0, &next_state_tensor);
            let next_q_token = next_q_args.request_fetch(&self.output_op, 0);
            self.session.run(&mut next_q_args)?;
            let next_q_values: Tensor<f32> = next_q_args.fetch(next_q_token)?;
            
            let max_next_q = next_q_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            
            let target_q = if experience.done {
                experience.reward
            } else {
                experience.reward + discount_factor * max_next_q
            };
            target_qs_vec.push(target_q);
        }

        let states_tensor = Tensor::new(&[batch.len() as u64, (states_vec.len() / batch.len()) as u64]).with_values(&states_vec)?;
        let actions_tensor = Tensor::new(&[batch.len() as u64, 2]).with_values(&actions_vec.into_iter().flatten().collect::<Vec<i32>>())?;
        let target_qs_tensor = Tensor::new(&[batch.len() as u64]).with_values(&target_qs_vec)?;

        let mut train_args = SessionRunArgs::new();
        train_args.add_feed(&self.input_op, 0, &states_tensor);
        train_args.add_feed(&self.action_op, 0, &actions_tensor);
        train_args.add_feed(&self.target_q_op, 0, &target_qs_tensor);
        train_args.add_target(&self.train_op);
        
        self.session.run(&mut train_args)?;
        
        Ok(())
    }

    pub fn decide(&self, game_state: &GameState) -> i32 {
        let input_tensor = self.prepare_input(game_state);

        let mut args = SessionRunArgs::new();
        let output_token = args.request_fetch(&self.output_op, 0);
        args.add_feed(&self.input_op, 0, &input_tensor);

        if let Err(e) = self.session.run(&mut args) {
            eprintln!("Error running session: {:?}", e);
            return 0;
        }

        let output_tensor: Tensor<f32> = match args.fetch(output_token) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Error fetching output: {:?}", e);
                return 0;
            }
        };
        
        let output_slice = &output_tensor;
        
        let action = output_slice
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap_or(1);

        action as i32 - 1
    }

    fn prepare_input(&self, game_state: &GameState) -> Tensor<f32> {
        let mut features = Vec::new();
        
        // Player data
        // Player's horizontal position on screen, normalized.
        features.push(game_state.player.x / (super::WIDTH as f32) * 2.0 - 1.0);
        // Player's vertical position on screen, normalized.
        features.push((game_state.player.y - game_state.camera_y) / (super::HEIGHT as f32) * 2.0 - 1.0);
        // Player's vertical velocity, clamped and normalized.
        features.push(game_state.player.vy.max(-5.0).min(5.0) / 5.0);

        let mut sorted_platforms = game_state.platforms.clone();
        
        sorted_platforms.sort_by(|a, b| {
            let dist_a = (a.x as f32 - game_state.player.x).powi(2) + (a.y - game_state.player.y).powi(2);
            let dist_b = (b.x as f32 - game_state.player.x).powi(2) + (b.y - game_state.player.y).powi(2);
            dist_a.partial_cmp(&dist_b).unwrap()
        });

        for p in sorted_platforms.iter().take(8) {
            // Platform positions relative to the player, normalized.
            features.push((p.x as f32 - game_state.player.x) / (super::WIDTH as f32));
            features.push((p.y - game_state.player.y) / (super::HEIGHT as f32));
        }

        let feature_count = 3 + 2 * 8;
        while features.len() < feature_count {
            features.push(0.0);
        }
        
        features.truncate(feature_count);

        Tensor::new(&[1, feature_count as u64])
            .with_values(&features)
            .unwrap()
    }

    pub fn get_feature_vector(&self, game_state: &GameState) -> Vec<f32> {
        let tensor = self.prepare_input(game_state);
        tensor.to_vec()
    }
} 