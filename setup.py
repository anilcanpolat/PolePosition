from setuptools import setup

setup(
    name='PolePosition',
    version='0.1.0',
    install_requires=['gymnasium', 'pygame', 'numpy'],  # Add other dependencies as needed
    packages=['PolePosition'],  # Replace with your package directory
    entry_points={
        'gymnasium.envs': [
            'PolePosition-v0 = PolePosition.envs:PolePositionEnv',
            # 'PolePosition-pixels-v0 = PolePosition.envs:PolePositionEnv'  # If you have different configurations, you can register them separately
            #print("testesttest")
        ]
    }
)