# Super Mario AI: Reinforcement Learning with PPO

## Project Overview
This project demonstrates the implementation of a **Proximal Policy Optimization (PPO)** algorithm to train an AI agent to autonomously play **Super Mario**. The AI utilizes reinforcement learning techniques to navigate through levels, avoiding obstacles and enemies, while maximizing its score.

### Key Features
- **Game Environment**: Trained using the **SuperMarioBros-v0** environment.
- **Algorithm**: Implemented PPO for efficient policy optimization.
- **Preprocessing**: Used gray-scaled frame stacking for better performance and faster convergence.
- **Outcome**: The AI successfully learned to play the game, showcasing adaptability and strategy.

---

## Demo
Watch the AI in action!

[![Super Mario Gameplay](https://img.youtube.com/vi/KhsmDC6bTX8/0.jpg)](https://www.youtube.com/watch?v=KhsmDC6bTX8)

> *Click the image above to watch the video on YouTube.*

---
## How to Run the Project
### Prerequisites
1. Python 3.8 or above.
2. Install dependencies:
   ```bash
   pip install gym[all] stable-baselines3 opencv-python
## Results
The PPO algorithm demonstrated robust performance, allowing the agent to:

Navigate obstacles and enemies with minimal failures.
Achieve higher scores consistently across training iterations.

## Acknowledgments
Special thanks to Gym for the game environment and Stable-Baselines3 for the reinforcement learning toolkit.


Here’s a sample GitHub Markdown file for your project, including a video playback of "SuperMarioBros-v0 2024-12-26 16-29-39.mp4":

markdown
Copy code
# Super Mario AI: Reinforcement Learning with PPO

## Project Overview
This project demonstrates the implementation of a **Proximal Policy Optimization (PPO)** algorithm to train an AI agent to autonomously play **Super Mario**. The AI utilizes reinforcement learning techniques to navigate through levels, avoiding obstacles and enemies, while maximizing its score.

### Key Features
- **Game Environment**: Trained using the **SuperMarioBros-v0** environment.
- **Algorithm**: Implemented PPO for efficient policy optimization.
- **Preprocessing**: Used gray-scaled frame stacking for better performance and faster convergence.
- **Outcome**: The AI successfully learned to play the game, showcasing adaptability and strategy.

---
## About the PPO Algorithm

**Proximal Policy Optimization (PPO)** is a state-of-the-art reinforcement learning algorithm known for its balance between simplicity and performance. It is widely used in applications such as game AI, robotics, and simulations.

### How PPO Works

1. **Actor-Critic Framework**:
   PPO uses an actor-critic architecture:
   - **Actor**: Predicts the next action based on the current policy.
   - **Critic**: Evaluates the action taken by estimating the value function.

2. **Clip Objective**:
   - PPO optimizes the policy by limiting large updates to avoid drastic changes.
   - The **clip function** ensures the new policy stays close to the old one by introducing a constraint:
     \[
     L^{CLIP} = \min\left(r(\theta)A, \text{clip}\left(r(\theta), 1-\epsilon, 1+\epsilon\right)A\right)
     \]
     where:
     - \( r(\theta) \): Probability ratio of new vs. old policy.
     - \( \epsilon \): Clipping threshold (e.g., 0.2).
     - \( A \): Advantage estimation.

3. **Advantage Estimation**:
   PPO uses Generalized Advantage Estimation (GAE) to compute \( A \), which balances bias and variance in the learning process.

4. **Entropy Regularization**:
   - Encourages exploration by adding an entropy term to the loss function.
   - Helps avoid premature convergence to suboptimal policies.

5. **Minibatch Training**:
   - PPO trains over multiple minibatches of experience data, improving sample efficiency.

---

### Why PPO is Effective
- **Stability**: The clipped objective avoids large policy updates, ensuring stable learning.
- **Efficiency**: Combines the benefits of trust region optimization with ease of implementation.
- **Versatility**: Performs well across diverse environments.

---

### Example: PPO in Super Mario

In this project, PPO was applied to train an AI agent to play **Super Mario** using the **SuperMarioBros-v0** environment. Here's a breakdown of its application:

1. **Environment Interaction**:
   - The agent observes the game state as frames.
   - Actions are selected based on the policy (e.g., move right, jump).

2. **Reward Function**:
   - Rewards are provided for progressing through the level and defeating enemies.
   - Negative rewards (penalties) are given for collisions or falling.

3. **Policy Update**:
   - After interacting with the environment for several steps, PPO optimizes the policy using collected trajectories.
   - Updates are clipped to ensure smooth policy improvement.

---

### Results in Super Mario
- **Training**: The agent learned to:
  - Navigate levels efficiently.
  - Time jumps to avoid gaps and enemies.
- **Performance**: After training, the AI achieved a higher success rate and adapted to different level challenges.

---

### Other Applications of PPO
1. **Robotics**: Training robots to perform complex tasks such as picking objects or walking.
2. **Games**: Developing AI for real-time strategy games or complex board games like Go.
3. **Simulations**: Autonomous driving or flight control in simulations.

---

## Further Reading
- [Original PPO Paper](https://arxiv.org/abs/1707.06347)
- [Stable-Baselines3 PPO Implementation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [OpenAI Spinning Up Guide on PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

--- 

## Demo
Watch the AI in action!

![Super Mario Gameplay](SuperMarioBros-v0%202024-12-26%2016-29-39.mp4)

> *Note: Ensure the video file is uploaded to the repository to enable playback.*

---

## How to Run the Project
### Prerequisites
1. Python 3.8 or above.
2. Install dependencies:
   ```bash
   pip install gym[all] stable-baselines3 opencv-python
Run the Training
bash
Copy code
python train.py
Evaluate the Model
bash
Copy code
python evaluate.py
Results
The PPO algorithm demonstrated robust performance, allowing the agent to:

Navigate obstacles and enemies with minimal failures.
Achieve higher scores consistently across training iterations.
Repository Structure
bash
Copy code
.
├── train.py        # Training script for the PPO algorithm
├── evaluate.py     # Script to evaluate the trained model
├── models/         # Saved models
├── media/          # Video and image assets
└── README.md       # Project documentation
Acknowledgments
Special thanks to Gym for the game environment and Stable-Baselines3 for the reinforcement learning toolkit.

Author
Abishek Ravi

Feel free to connect or contribute to this project!
