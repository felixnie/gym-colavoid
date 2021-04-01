import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# LiDAR + discrete action space
register(
    id='ColAvoid-v0',
    entry_point='gym_colavoid.envs:ColAvoidEnvDiscrete',
)

# Simplified measurement + continuous action space
register(
    id='ColAvoid-v1',
    entry_point='gym_colavoid.envs:ColAvoidEnv',
)
