import time
import random
import json
from malmo.MalmoPython import *
import matplotlib.pyplot as plt

# 初始化 Agent
agent = AgentHost()
try:
    agent.parse([])
except RuntimeError:
    pass

# 動作設定（含細轉角度）
actions = ["move 1", "turn -0.5", "turn 0.5", "move 0"]
q_table = {}
epsilon = 0.3
alpha = 0.5
gamma = 0.9

# 地圖與任務設定
mission_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <About>
    <Summary>學會轉彎任務</Summary>
  </About>
  <ServerSection>
    <ServerHandlers>
      <FlatWorldGenerator generatorString="3;7,2;1;"/>
      <DrawingDecorator>
        <!-- 垂直道路 -->
        <DrawLine x1="0" y1="4" z1="0" x2="0" y2="4" z2="8" type="coal_block"/>
        <!-- 左轉入口點補上 coal_block 引導 -->
        <DrawBlock x="-1" y="4" z="8" type="coal_block"/>
        <!-- 左轉轉角提示 -->
        <DrawBlock x="-1" y="4" z="9" type="gold_block"/>
        <DrawBlock x="-2" y="4" z="8" type="gold_block"/>
        <!-- 左轉後延伸 -->
        <DrawLine x1="-2" y1="4" z1="8" x2="-5" y2="4" z2="8" type="coal_block"/>
        <DrawLine x1="-5" y1="4" z1="8" x2="-5" y2="4" z2="12" type="coal_block"/>
        <!-- 終點 -->
        <DrawBlock x="-5" y="4" z="13" type="diamond_block"/>

        <!-- 石牆 -->
        <DrawCuboid x1="-1" y1="4" z1="8" x2="1" y2="4" z2="8" type="stone"/>
        <DrawCuboid x1="-1" y1="4" z1="0" x2="-1" y2="4" z2="8" type="stone"/>
        <DrawCuboid x1="1" y1="4" z1="0" x2="1" y2="4" z2="8" type="stone"/>
        <DrawCuboid x1="-5" y1="4" z1="7" x2="0" y2="4" z2="7" type="stone"/>
        <DrawCuboid x1="-6" y1="4" z1="8" x2="-6" y2="4" z2="13" type="stone"/>
        <DrawCuboid x1="-4" y1="4" z1="8" x2="-4" y2="4" z2="13" type="stone"/>
      </DrawingDecorator>
      <ServerQuitWhenAnyAgentFinishes/>
    </ServerHandlers>
  </ServerSection>

  <AgentSection mode="Survival">
    <Name>TurnTrainer</Name>
    <AgentStart>
      <Placement x="0.5" y="5" z="0.5" yaw="0"/>
    </AgentStart>
    <AgentHandlers>
      <ObservationFromGrid>
        <Grid name="floor5x5">
          <min x="-2" y="-1" z="1"/>
          <max x="2" y="-1" z="5"/>
        </Grid>
      </ObservationFromGrid>
      <ObservationFromFullStats/>
      <RewardForTouchingBlockType>
        <Block type="coal_block" reward="0.2"/>
        <Block type="gold_block" reward="6.0"/>
        <Block type="stone" reward="-20.0"/>
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

def start_mission():
    mission = MissionSpec(mission_xml, True)
    record = MissionRecordSpec()
    for retry in range(3):
        try:
            agent.startMission(mission, record)
            break
        except RuntimeError as e:
            print(f"🛑 任務啟動失敗原因：{e}")
            if retry == 2:
                return False
            time.sleep(2)
    print("⌛ 等待任務開始...")
    world_state = agent.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="", flush=True)
        time.sleep(0.5)
        world_state = agent.getWorldState()
    print("\n✅ 任務開始！")
    return True

def get_state(world_state):
    if len(world_state.observations) == 0:
        return "unknown"
    obs = json.loads(world_state.observations[-1].text)
    grid = obs.get("floor5x5", ["air"] * 25)
    indices = [6,7,8,11,12,13,16,17,18]
    yaw = round(obs.get("Yaw", 0), -1)
    return (tuple(grid[i] for i in indices), yaw)

def choose_action(state):
    if random.random() < epsilon or state not in q_table:
        return random.choice(actions)
    return max(q_table[state], key=q_table[state].get)

def update_q_table(state, action, reward, next_state):
    if state not in q_table:
        q_table[state] = {a: 0.0 for a in actions}
    if next_state not in q_table:
        q_table[next_state] = {a: 0.0 for a in actions}
    old_q = q_table[state][action]
    next_max = max(q_table[next_state].values())
    q_table[state][action] = old_q + alpha * (reward + gamma * next_max - old_q)

# 主訓練迴圈
episode_rewards = []
num_episodes = 100
for episode in range(num_episodes):
    print(f"\n=== Episode {episode+1}/{num_episodes} ===")
    if episode % 20 == 0 and episode > 0:
        epsilon = max(0.05, epsilon * 0.95)
        print(f"🔁 探索率降低為：{epsilon:.2f}")

    if not start_mission():
        continue

    total_reward = 0
    start_time = time.time()

    while agent.getWorldState().is_mission_running:
        world_state = agent.getWorldState()
        if world_state.number_of_observations_since_last_state > 0:
            state = get_state(world_state)
            action = choose_action(state)
            agent.sendCommand(action)
            time.sleep(0.05)
            world_state = agent.getWorldState()
            reward = sum(r.getValue() for r in world_state.rewards)
            total_reward += reward
            if world_state.number_of_observations_since_last_state > 0:
                next_state = get_state(world_state)
                update_q_table(state, action, reward, next_state)

        if time.time() - start_time > 15:
            print("⌛ 回合超時")
            break

    try:
        agent.sendCommand("move 0")
        agent.sendCommand("turn 0")
        time.sleep(0.5)
    except:
        pass

    while agent.getWorldState().is_mission_running:
        time.sleep(0.1)
    time.sleep(1)

    if total_reward >= 10:
        print(f"✅ 第 {episode+1} 輪成功到達終點！（Reward: {total_reward:.2f}）")

    episode_rewards.append(total_reward)
    print(f"🎯 Episode {episode+1} 完成，Reward: {total_reward:.2f}")

def plot_rewards(episode_rewards):
    plt.figure(figsize=(10, 4))
    plt.plot(episode_rewards, label='Reward per episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Curve: Q-Learning Left Turn')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
plot_rewards(episode_rewards)