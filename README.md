# Pole Position Game

Welcome to the Pole Position Game! This is a simple racing game implemented using Pygame and the Gymnasium library. The objective is to complete laps on a circular track while navigating through checkpoints.

## Game Overview

In this game, you control a car on a circular track. The goal is to complete laps by passing through specific checkpoints in order. The game terminates after completing the required number of laps.

## Features

- Circular track with a defined radius and width.
- Realistic car physics with acceleration, deceleration, and rotation.
- Five proximity rays to detect distances to the track boundaries.
- Checkpoints and lap completion logic.
- Two observation modes: feature-based and pixel-based.
- Rendering options: human-readable display and RGB array.

## Getting Started

### Prerequisites

- Python 3.6+
- Pygame
- Gymnasium

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/TU-Dortmund-Fachprojekt-RL-SS24/anilcanpolat.git
    cd pole-position-game
    ```

2. Install the required dependencies:
    ```bash
    pip install pygame gymnasium numpy
    ```

### Running the Game

To run the game, execute the following command:

```bash
python main.py
