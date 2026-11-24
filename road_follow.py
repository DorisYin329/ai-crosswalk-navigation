import time
import random
import json
from malmo.MalmoPython import *

# 初始化 Agent
agent = AgentHost()
agent.parse(["road_follow.py"])  # 避免 vector<T> too long

# 設定簡單 Q-Table
q_table = {}
actions = ["move 1", "turn -1", "turn 1", "move 0"]

epsilon = 0.1  # 探索率
alpha = 0.5    # 學習率
gamma = 0.9    # 折扣因子

# 任務 XML
mission_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <About>
    <Summary>簡易道路辨識</Summary>
  </About>
  <ServerSection>
    <ServerHandlers>
      <FlatWorldGenerator generatorString="3;7,2;1;"/>
      <DrawingDecorator>
        <DrawLine x1="0" y1="4" z1="0" x2="0" y2="4" z2="19" type="coal_block"/>
        <DrawBlock x="0" y="4" z="20" type="diamond_block"/>
        <DrawCuboid x1="-1" y1="4" z1="0" x2="-1" y2="4" z2="20" type="stone"/>
        <DrawCuboid x1="1" y1="4" z1="0" x2="1" y2="4" z2="20" type="stone"/>
      </DrawingDecorator>
      <ServerQuitWhenAnyAgentFinishes/>
    </ServerHandlers>
  </ServerSection>
  <AgentSection mode="Survival">
    <Name>RoadFollower</Name>
    <AgentStart>
      <Placement x="0.5" y="5" z="0.5" yaw="0"/>
    </AgentStart>
    <AgentHandlers>
      <ObservationFromGrid>
        <Grid name="floor3x3">
          <min x="-1" y="-1" z="1"/>
          <max x="1" y="-1" z="3"/>
        </Grid>
      </ObservationFromGrid>
      <RewardForTouchingBlockType>
        <Block type="coal_block" reward="1.0"/>
        <Block type="stone" reward="-10.0"/>
        <Block type="diamond_block" reward="10.0"/>
      </RewardForTouchingBlockType>
      <ContinuousMovementCommands/>
      <AgentQuitFromTouchingBlockType>
        <Block type="stone"/>
        <Block type="diamond_block"/>
      </AgentQuitFromTouchingBlockType>
    </AgentHandlers>
  </AgentSection>
</Mission>'''

# 啟動任務
def start_mission():
    mission = MissionSpec(mission_xml, True)
    record = MissionRecordSpec()
    max_retries = 3
    for retry in range(max_retries):
        try:
            agent.startMission(mission, record)
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("❌ 啟動失敗:", e)
                exit(1)
            else:
                time.sleep(2)

    print("⌛ 等待任務開始...")
    world_state = agent.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="", flush=True)
        time.sleep(0.5)
        world_state = agent.getWorldState()
    print("\n✅ 任務開始！")

# 取得 state
def get_state(world_state):
    if len(world_state.observations) == 0:
        return "unknown"
    obs_text = world_state.observations[-1].text
    obs = json.loads(obs_text)
    grid = obs.get("floor3x3", ["air"] * 9)
    return tuple(grid)

# 選擇 action
def choose_action(state):
    if random.random() < epsilon or state not in q_table:
        return random.choice(actions)
    else:
        return max(q_table[state], key=q_table[state].get)

# 更新 Q-Table
def update_q_table(state, action, reward, next_state):
    if state not in q_table:
        q_table[state] = {a: 0.0 for a in actions}
    if next_state not in q_table:
        q_table[next_state] = {a: 0.0 for a in actions}

    old_q = q_table[state][action]
    next_max = max(q_table[next_state].values())
    new_q = old_q + alpha * (reward + gamma * next_max - old_q)
    q_table[state][action] = new_q

# 主訓練回圈
num_episodes = 20

for episode in range(num_episodes):
    print(f"\n=== Episode {episode+1}/{num_episodes} ===")
    start_mission()

    total_reward = 0
    episode_start_time = time.time()

    while agent.getWorldState().is_mission_running:
        world_state = agent.getWorldState()

        if world_state.number_of_observations_since_last_state > 0:
            state = get_state(world_state)
            action = choose_action(state)
            agent.sendCommand(action)

            time.sleep(0.05)  # 等世界反應

            world_state = agent.getWorldState()

            # 收 reward
            reward = sum(r.getValue() for r in world_state.rewards)
            total_reward += reward

            if world_state.number_of_observations_since_last_state > 0:
                next_state = get_state(world_state)
                update_q_table(state, action, reward, next_state)

        # 額外超時保護
        if time.time() - episode_start_time > 10:
            print("⌛ 超過10秒，強制結束本回合")
            break

    # 🛑 任務結束時，主動讓 agent 停止行動，避免TCP錯誤
    try:
        agent.sendCommand("move 0")
        agent.sendCommand("turn 0")
        time.sleep(0.5)
    except:
        pass  # 任務已經結束就忽略

    print(f"🎯 Episode {episode+1} 結束，總 reward: {total_reward:.2f}")