\section{Background and Related Work}

\subsection{Reinforcement Learning Fundamentals}

Reinforcement learning addresses the problem of sequential decision-making under uncertainty, where an agent learns to maximize cumulative rewards by interacting with its environment. The standard formulation models this interaction as a Markov Decision Process (MDP), characterized by states, actions, transition probabilities, and rewards. At each time step, the agent observes a state, selects an action, receives a reward, and transitions to a new state. The agent's objective is to learn a policy—a mapping from states to actions—that maximizes the expected cumulative discounted reward over time~\cite{sutton2018reinforcement}.

Value-based methods, a prominent family of reinforcement learning algorithms, estimate the expected return for each state-action pair. Q-learning, in particular, learns an action-value function $Q(s,a)$ that represents the expected return when taking action $a$ in state $s$ and following the optimal policy thereafter. The Q-function is updated iteratively using the Bellman equation, which relates the value of a state-action pair to the expected immediate reward and the values of subsequent state-action pairs.

\subsection{Deep Q-Networks}

Traditional Q-learning becomes impractical in environments with large or continuous state spaces due to the curse of dimensionality. Deep Q-Networks (DQN) address this limitation by approximating the Q-function using deep neural networks. The seminal work by Mnih et al.~\cite{mnih2015human} demonstrated that deep reinforcement learning could achieve human-level performance on Atari games, learning directly from raw pixel inputs.

Several key innovations enable stable learning in DQN. Experience replay stores transitions in a buffer and samples mini-batches randomly for training, breaking correlations between consecutive samples and improving data efficiency. Target networks provide stable learning targets by periodically copying the weights from the primary network, reducing the moving target problem inherent in bootstrapped learning.

Recent advances have significantly improved upon the original DQN algorithm. Dabney et al.~\cite{dabney2020distributional} introduced distributional reinforcement learning, which models the entire distribution of returns rather than just their expected values, leading to more robust learning and improved performance. Kumar et al.~\cite{kumar2023offline} addressed fundamental barriers in offline reinforcement learning, enabling agents to learn from pre-collected datasets without additional environment interaction. This approach is particularly valuable for applications where exploration is costly or risky.

\subsection{Reinforcement Learning in Game Environments}

Game environments serve as valuable testbeds for reinforcement learning research due to their controlled nature, clear objectives, and scalable complexity. Significant milestones include AlphaGo's mastery of the ancient game of Go~\cite{silver2017mastering}, Agent57's superhuman performance across all 57 Atari games~\cite{badia2020agent57}, and AlphaStar's grandmaster-level play in the complex real-time strategy game StarCraft II~\cite{vinyals2019grandmaster}.

The field has recently witnessed a paradigm shift toward model-based approaches, where agents learn explicit models of their environments to aid planning and decision-making. Hafner et al.~\cite{hafner2023mastering} demonstrated that world models enable agents to master diverse domains through imagination-based planning, while Yu et al.~\cite{yu2022planning} developed diffusion models for flexible behavior synthesis in complex environments.

Transformer-based architectures have emerged as a powerful framework for reinforcement learning, recasting RL as a sequence prediction problem. Chen et al.~\cite{chen2021decision} introduced Decision Transformers, which frame reinforcement learning as conditional sequence modeling, while Lee et al.~\cite{lee2022multi} extended this approach to multi-game scenarios, demonstrating impressive transfer learning capabilities.

\subsection{Deep Reinforcement Learning for Flappy Bird}

Flappy Bird presents an interesting challenge for reinforcement learning due to its deceptively simple mechanics combined with the need for precise timing and control. Prior work has applied various RL approaches to this domain with mixed results. While several implementations have demonstrated successful agents, most focus on either very simple Q-learning approaches with discretized state spaces or complex architectures that require substantial computational resources.

Our work builds upon these foundations while focusing on a minimalist yet effective approach. We explore the balance between state representation complexity and learning efficiency, aiming to develop an agent that learns effectively while remaining computationally tractable. Unlike approaches that use raw pixels as input, we employ a compact state representation that captures the essential information for decision-making, inspired by recent work on state abstraction in reinforcement learning~\cite{lee2022multi}.

Furthermore, we draw inspiration from the latest innovations in reinforcement learning algorithms, particularly those focused on sample efficiency and stability. Our implementation incorporates elements from distributional reinforcement learning and modern approaches to exploration, resulting in a system that learns more effectively from limited experience. This aligns with current research trends toward making reinforcement learning more practical for real-world applications~\cite{wang2022offline}.