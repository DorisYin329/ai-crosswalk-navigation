import time
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from malmo.MalmoPython import *
import sys
import ast # <<<--- 新增導入 ast 模塊

# 初始化 Agent
agent = AgentHost()
try:
    agent.parse(sys.argv)
except RuntimeError as e:
    print('ERROR:', e)
    print(agent.getUsage())
    exit(1)
if agent.receivedArgument("help"):
    print(agent.getUsage())
    exit(0)

# --- Q-table 文件名 ---
# 您可以更改此文件名，或者在運行時通過參數傳入
# 設為 None 則不加載，並在保存時生成帶時間戳的文件名
Q_TABLE_FILENAME = "q-table_直線.json"
# Q_TABLE_FILENAME = None # 如果不想加載特定文件，或者首次運行

# Q-learning 設定
q_table = {} # 初始化為空字典
actions = ["move 1", "turn -1", "turn 1", "move 0"]
initial_epsilon = 0.3
min_epsilon = 0.01 # 您更新的值
epsilon_decay_rate = 0.995
epsilon = initial_epsilon
alpha = 0.1
gamma = 0.9
TERMINAL_STATE = "TERMINAL"

VISITED_PENALTY = -0.6
STEP_PENALTY = -0.03
STAY_ON_COAL_REWARD = 0.4
NO_PROGRESS_PENALTY = -0.25

# --- 嘗試加載 Q-table ---
if Q_TABLE_FILENAME:
    try:
        print(f"\n💾 正在從 {Q_TABLE_FILENAME} 加載 Q-table...")
        with open(Q_TABLE_FILENAME, 'r') as f:
            loaded_q_table_str_keys = json.load(f)
        
        q_table_from_file = {}
        for str_state_key, actions_dict in loaded_q_table_str_keys.items():
            if str_state_key == TERMINAL_STATE or str_state_key == "unknown":
                state_key = str_state_key
            elif str_state_key.startswith("(") and str_state_key.endswith(")"):
                try:
                    state_key = ast.literal_eval(str_state_key)
                    if not isinstance(state_key, tuple):
                        print(f"Warning: Key '{str_state_key}' after eval is not a tuple. Treating as string.")
                        state_key = str_state_key
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Could not convert key '{str_state_key}' back to tuple: {e}. Treating as string key.")
                    state_key = str_state_key
            else:
                state_key = str_state_key
            
            q_table_from_file[state_key] = actions_dict
        
        q_table = q_table_from_file
        print(f"✅ Q-table 已成功加載。Q-table 大小: {len(q_table)}")
    except FileNotFoundError:
        print(f"ℹ️ 未找到 Q-table 文件 {Q_TABLE_FILENAME}。將從空的 Q-table 開始訓練。")
    except Exception as e:
        print(f"❌ 加載 Q-table 失敗: {e}。將從空的 Q-table 開始訓練。")
else:
    print("ℹ️ 未指定 Q-table 加載文件。將從空的 Q-table 開始訓練。")


# XML 任務設計 (道路保持縮短，獎勵值更新)
mission_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <About><Summary>road_follow_new_rewards_v2</Summary></About>
  <ServerSection>
    <ServerHandlers>
      <FlatWorldGenerator generatorString="3;7,2;1;"/>
      <DrawingDecorator>
        <DrawCuboid x1="-1" y1="1" z1="0" x2="1" y2="1" z2="19" type="coal_block"/>
        <DrawBlock x="0" y="1" z="20" type="diamond_block"/>
        <DrawBlock x="1" y="1" z="20" type="diamond_block"/>
        <DrawBlock x="-1" y="1" z="20" type="diamond_block"/>
        <DrawCuboid x1="-2" y1="1" z1="0" x2="-2" y2="1" z2="20" type="stone"/>
        <DrawCuboid x1="2" y1="1" z1="0" x2="2" y2="1" z2="20" type="stone"/>
        <DrawCuboid x1="3" y1="1" z1="0" x2="3" y2="1" z2="20" type="stone"/>
        <DrawCuboid x1="-3" y1="1" z1="0" x2="-3" y2="1" z2="20" type="stone"/>
      </DrawingDecorator>
      <ServerQuitWhenAnyAgentFinishes/>
    </ServerHandlers>
  </ServerSection>
  <AgentSection mode="Survival">
    <Name>RoadFollower</Name>
    <AgentStart><Placement x="0.5" y="2.0" z="0.5" yaw="0"/></AgentStart>
    <AgentHandlers>
      <ObservationFromGrid>
        <Grid name="floor5x5">
          <min x="-2" y="-1" z="1"/>
          <max x="2" y="-1" z="5"/>
        </Grid>
      </ObservationFromGrid>
      <RewardForTouchingBlockType>
        <Block type="coal_block" reward="5.0"/>      <Block type="stone" reward="-0.5"/>     <Block type="diamond_block" reward="100.0"/>   <Block type="grass" reward="0.0"/>        </RewardForTouchingBlockType>
      <ContinuousMovementCommands/>
      <AgentQuitFromTouchingBlockType>
        <Block type="grass"/>
        <Block type="diamond_block"/>
      </AgentQuitFromTouchingBlockType>
      <AgentQuitFromTimeUp timeLimitMs="45000" description="time_up"/>
    </AgentHandlers>
  </AgentSection>
</Mission>'''

def start_mission():
    mission = MissionSpec(mission_xml, True)
    record = MissionRecordSpec()
    for retry in range(3):
        try:
            agent.startMission(mission, record)
            break
        except RuntimeError as e:
            if retry == 2:
                print("❌ 啟動失敗:", e)
                exit(1)
            time.sleep(2)

    print("⌛ 等待任務開始...")
    world_state = agent.getWorldState()
    while not world_state.has_mission_begun:
        time.sleep(0.1)
        world_state = agent.getWorldState()
        for error in world_state.errors: print("Error during mission start wait:",error.text)
        if not world_state.is_mission_running and world_state.has_mission_begun :
            print("❌ 任務在開始前異常終止。")
            return False
    print("✅ 任務開始！")
    return True

def get_state(world_state):
    if not world_state.is_mission_running and len(world_state.observations) == 0:
        return TERMINAL_STATE
    if len(world_state.observations) > 0:
        try:
            obs_text = world_state.observations[-1].text
            obs = json.loads(obs_text)
            grid_obs = obs.get("floor5x5", ["air"] * 25)
            if not isinstance(grid_obs, list) or len(grid_obs) != 25:
                grid_obs = ["air"] * 25
            return tuple(grid_obs)
        except json.JSONDecodeError as e:
            return "unknown"
        except Exception as e:
            return "unknown"
    return "unknown"

def get_reward_from_environment(world_state):
    reward_val = 0.0
    if world_state.rewards:
        for r in world_state.rewards:
            reward_val += r.getValue()
    return reward_val

def choose_action(state):
    if state == TERMINAL_STATE or state == "unknown":
        return "move 0"
    # 在加載 Q-table 後，如果 epsilon 已經很低，可能仍然想在初期強制一些探索
    # current_epsilon = epsilon if episode > WARM_UP_EPISODES else initial_epsilon 
    # 但這裡我們保持簡單，直接使用全局的 epsilon
    if random.random() < epsilon or state not in q_table:
        return random.choice(actions)
    else:
        if q_table.get(state): # 使用 .get 避免 state 剛好是新遇到的但 epsilon 很低的情況
            return max(q_table[state], key=q_table[state].get)
        else: # 理論上如果 state in q_table 為 false, 這裡不會執行
            return random.choice(actions)


def update_q_table(state, action, reward, next_state):
    if state in [TERMINAL_STATE, "unknown"]:
        return

    if state not in q_table:
        q_table[state] = {a: 0.0 for a in actions}

    if next_state != TERMINAL_STATE and next_state != "unknown" and next_state not in q_table:
        q_table[next_state] = {a: 0.0 for a in actions}

    old_q_value = q_table[state].get(action, 0.0)

    if next_state == TERMINAL_STATE or next_state == "unknown" or not q_table.get(next_state):
        next_max_q = 0.0
    else:
        if q_table[next_state]: # 確保 next_state 在 q_table 中有條目且其動作字典非空
             next_max_q = max(q_table[next_state].values())
        else: # 理論上如果 next_state 被正確初始化，這裡不應發生
            next_max_q = 0.0

    q_table[state][action] = old_q_value + alpha * (reward + gamma * next_max_q - old_q_value)

# 主訓練流程
num_episodes = 1000
reward_log = []
timesteps_log = []

DIAMOND_REWARD_FROM_XML = 100.0
# GRASS_XML_REWARD = 0.0

# 是否在每次訓練後覆蓋 Q-table，或者只在最後保存
SAVE_Q_TABLE_ON_FINISH_ONLY = True # True: 只在所有episodes結束後保存一次; False: 每次都嘗試保存最新

for episode in range(num_episodes):
    print(f"\n=== Episode {episode+1}/{num_episodes}, Epsilon: {epsilon:.3f} ===")
    if not start_mission():
        print("Skipping episode due to mission start failure.")
        time.sleep(5)
        continue

    current_world_state = agent.getWorldState()
    while current_world_state.is_mission_running and current_world_state.number_of_observations_since_last_state == 0:
        time.sleep(0.05)
        current_world_state = agent.getWorldState()
        for error in current_world_state.errors: print("Error during initial observation wait:", error.text)

    state = get_state(current_world_state)

    total_episode_reward = 0
    step_count = 0
    visited_states_in_episode = set()

    if state != TERMINAL_STATE and state != "unknown":
        visited_states_in_episode.add(state)

    block_in_front_center_prev_step = "air"

    while current_world_state.is_mission_running:
        step_count += 1

        if state != TERMINAL_STATE and state != "unknown" and isinstance(state, tuple) and len(state) == 25:
            block_in_front_center_prev_step = state[2]
        else:
            block_in_front_center_prev_step = "air"

        action = choose_action(state)
        agent.sendCommand(action)
        time.sleep(0.1)

        world_state_after_action = agent.getWorldState()
        observation_attempts = 0
        while world_state_after_action.is_mission_running and \
              world_state_after_action.number_of_observations_since_last_state == 0 and \
              observation_attempts < 10:
            time.sleep(0.05)
            world_state_after_action = agent.getWorldState()
            observation_attempts+=1

        reward_from_env = get_reward_from_environment(world_state_after_action)
        current_revisit_penalty = 0.0
        current_stay_on_coal_reward = 0.0
        current_no_progress_penalty = 0.0

        next_state = get_state(world_state_after_action)

        if action == "move 1" and block_in_front_center_prev_step == "coal_block":
            current_stay_on_coal_reward = STAY_ON_COAL_REWARD

        if next_state != TERMINAL_STATE and next_state != "unknown":
            if next_state in visited_states_in_episode:
                current_revisit_penalty