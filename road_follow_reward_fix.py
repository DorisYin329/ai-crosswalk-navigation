import time
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from malmo.MalmoPython import *

# 初始化 Agent
agent = AgentHost()
agent.parse(["road_follow.py"])

# Q-learning 設定
q_table = {}
actions = ["move 1", "turn -1", "turn 1", "move 0"]
initial_epsilon = 1.0
min_epsilon = 0.1
epsilon_decay_rate = 0.98
epsilon = initial_epsilon
alpha = 0.1
gamma = 0.9
TERMINAL_STATE = "TERMINAL"

# XML 任務設計
mission_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <About><Summary>擴大道路與觀察範圍</Summary></About>
  <ServerSection>
    <ServerHandlers>
      <FlatWorldGenerator generatorString="3;7,2;1;"/>
      <DrawingDecorator>
        <DrawCuboid x1="-1" y1="4" z1="0" x2="1" y2="4" z2="19" type="coal_block"/>
        <DrawBlock x="0" y="4" z="20" type="diamond_block"/>
        <DrawCuboid x1="-2" y1="4" z1="0" x2="-2" y2="4" z2="20" type="stone"/>
        <DrawCuboid x1="2" y1="4" z1="0" x2="2" y2="4" z2="20" type="stone"/>
      </DrawingDecorator>
      <ServerQuitWhenAnyAgentFinishes/>
    </ServerHandlers>
  </ServerSection>
  <AgentSection mode="Survival">
    <Name>RoadFollower</Name>
    <AgentStart><Placement x="0.5" y="5" z="0.5" yaw="0"/></AgentStart>
    <AgentHandlers>
      <ObservationFromGrid>
        <Grid name="floor5x5"><min x="-2" y="-1" z="1"/><max x="2" y="-1" z="5"/></Grid>
      </ObservationFromGrid>
      <RewardForTouchingBlockType>
        <Block type="coal_block" reward="5.0"/>
        <Block type="stone" reward="-2.0"/>
        <Block type="diamond_block" reward="10.0"/>
      </RewardForTouchingBlockType>
      <ContinuousMovementCommands/>
      <AgentQuitFromTouchingBlockType>
        <Block type="stone"/>
        <Block type="diamond_block"/>
      </AgentQuitFromTouchingBlockType>
      <AgentQuitFromTimeUp timeLimitMs="30000" description="time_up"/>
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
        time.sleep(0.5)
        world_state = agent.getWorldState()
        if not world_state.is_mission_running:
            print("❌ 任務在開始前終止。")
            return False
    print("✅ 任務開始！")
    return True

def get_state(world_state):
    if not world_state.is_mission_running and len(world_state.observations) == 0:
        return TERMINAL_STATE
    if len(world_state.observations) > 0:
        try:
            obs = json.loads(world_state.observations[-1].text)
            return tuple(obs.get("floor5x5", ["air"] * 25))
        except json.JSONDecodeError:
            return "unknown"
    return "unknown"

def get_reward(world_state):
    if not world_state.rewards:
        return 0.0
    latest_reward = world_state.rewards[-1].getValue()
    return min(latest_reward, 10.0) if latest_reward > 0 else max(latest_reward, -5.0)

def choose_action(state):
    if state == TERMINAL_STATE or state == "unknown":
        return "move 0"
    if random.random() < epsilon or state not in q_table:
        return random.choice(actions)
    return max(q_table[state], key=q_table[state].get)

def update_q_table(state, action, reward, next_state):
    if state in [TERMINAL_STATE, "unknown"]:
        return
    if state not in q_table:
        q_table[state] = {a: 0.0 for a in actions}
    if next_state != TERMINAL_STATE and next_state not in q_table:
        q_table[next_state] = {a: 0.0 for a in actions}
    next_max_q = 0.0 if next_state == TERMINAL_STATE else max(q_table[next_state].values())
    old_q = q_table[state][action]
    q_table[state][action] = old_q + alpha * (reward + gamma * next_max_q - old_q)

# 主訓練流程
num_episodes = 500
reward_log = []

for episode in range(num_episodes):
    print(f"\n=== Episode {episode+1}/{num_episodes} ===")
    if not start_mission():
        print("Skipping episode due to mission start failure.")
        time.sleep(5)
        continue

    current_world_state = agent.getWorldState()
    while current_world_state.is_mission_running and current_world_state.number_of_observations_since_last_state == 0:
        time.sleep(0.1)
        current_world_state = agent.getWorldState()

    state = get_state(current_world_state)
    total_episode_reward = 0
    step_count = 0

    while current_world_state.is_mission_running:
        step_count += 1
        action = choose_action(state)
        agent.sendCommand(action)
        time.sleep(0.1)
        world_state_after_action = agent.getWorldState()

        reward = get_reward(world_state_after_action)
        total_episode_reward += reward

        next_state = TERMINAL_STATE if not world_state_after_action.is_mission_running else get_state(world_state_after_action)
        update_q_table(state, action, reward, next_state)
        state = next_state
        current_world_state = world_state_after_action

    # 結束原因判斷
    reason = "正常結束"
    for msg in current_world_state.mission_control_messages:
        if msg.text == "time_up":
            reason = "⏰ 超時"
            break

    print(f"🎯 Episode {episode+1} 結束，總 reward: {total_episode_reward:.2f}, Steps: {step_count}, 原因: {reason}")
    reward_log.append(total_episode_reward)

    if epsilon > min_epsilon:
        epsilon *= epsilon_decay_rate
    time.sleep(1)

print("\n🎉 訓練完成！")

# 繪圖：總 reward 與移動平均
window_size = 20
moving_avg = np.convolve(reward_log, np.ones(window_size)/window_size, mode='valid')
plt.plot(reward_log, label="Total reward")
plt.plot(moving_avg, label="Moving Avg", linewidth=2)
plt.title("Total reward per episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.show()
