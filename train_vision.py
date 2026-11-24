import time
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from malmo.MalmoPython import *
import sys
import ast # <<<--- 新增導入 ast 模塊
import os
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--collect", action="store_true", help="啟用資料蒐集模式（用 Q-table 貪婪策略）")
parser.add_argument("--out", type=str, default="dataset_run1", help="資料輸出資料夾")
parser.add_argument("--save_every", type=int, default=2, help="每幾步存一次影像/標籤")
parser.add_argument("--use_vision", action="store_true", help="用視覺模型當狀態")
parser.add_argument("--ppo", action="store_true", help="使用 PPO 取代 Q-learning 策略")
args, unknown = parser.parse_known_args()


# 初始化 Agent
agent = AgentHost()
try:
    agent.parse([sys.argv[0]] + unknown)
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
initial_epsilon = 1.0
min_epsilon = 0.05 # 您更新的值
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
        <DrawCuboid x1="-1" y1="1" z1="0" x2="1" y2="1" z2="9" type="coal_block"/>
        <DrawBlock x="0" y="1" z="10" type="diamond_block"/>
        <DrawBlock x="1" y="1" z="10" type="diamond_block"/>
        <DrawBlock x="-1" y="1" z="10" type="diamond_block"/>
        <DrawCuboid x1="-2" y1="1" z1="0" x2="-2" y2="1" z2="10" type="stone"/>
        <DrawCuboid x1="2" y1="1" z1="0" x2="2" y2="1" z2="10" type="stone"/>
        <DrawCuboid x1="3" y1="1" z1="0" x2="3" y2="1" z2="10" type="stone"/>
        <DrawCuboid x1="-3" y1="1" z1="0" x2="-3" y2="1" z2="10" type="stone"/>
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
      <AgentQuitFromTimeUp timeLimitMs="30000" description="time_up"/>
      <VideoProducer want_depth="false">
        <Width>640</Width>
        <Height>480</Height>
      </VideoProducer>
    </AgentHandlers>
  </AgentSection>
</Mission>'''

# ===== PPO 最小可跑版 =====

class RolloutBuf:
    def __init__(self):
        self.obs, self.act, self.rew, self.logp, self.val, self.done = [],[],[],[],[],[]
    def add(self, o,a,r,lp,v,d):
        self.obs.append(np.array(o, dtype=np.float32))
        self.act.append(a); self.rew.append(r); self.logp.append(lp); self.val.append(v); self.done.append(d)
    def flush(self):
        out = (np.array(self.obs,np.float32),
               np.array(self.act,np.int64),
               np.array(self.rew,np.float32),
               np.array(self.logp,np.float32),
               np.array(self.val,np.float32),
               np.array(self.done,np.float32))
        self.obs,self.act,self.rew,self.logp,self.val,self.done = [],[],[],[],[],[]
        return out

class PPOPolicy(nn.Module):
    def __init__(self, obs_dim=25, n_act=4):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim,128), nn.Tanh(),
            nn.Linear(128,128), nn.Tanh()
        )
        self.pi = nn.Linear(128, n_act)
        self.v  = nn.Linear(128, 1)
    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h)

class PPOAgent:
    def __init__(self, obs_dim, n_act, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, epochs=4, mb=64, ent=0.01):
        super().__init__()
        self.net = PPOPolicy(obs_dim, n_act).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma, self.lam, self.clip, self.epochs, self.mb, self.ent = gamma, lam, clip, epochs, mb, ent

    def select_action(self, obs):
        o = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, v = self.net(o)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(v.item())

    def update(self, buf):
        obs, act, rew, logp_old, val, done = buf
        adv, ret = [], []
        gae, next_v = 0.0, 0.0
        for t in reversed(range(len(rew))):
            delta = rew[t] + self.gamma * (1-done[t]) * next_v - val[t]
            gae = delta + self.gamma * self.lam * (1-done[t]) * gae
            adv.append(gae); next_v = val[t]
            ret.append(gae + val[t])
        adv = torch.tensor(list(reversed(adv)), dtype=torch.float32, device=device)
        ret = torch.tensor(list(reversed(ret)), dtype=torch.float32, device=device)
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
        act = torch.tensor(act, dtype=torch.int64, device=device)
        logp_old = torch.tensor(logp_old, dtype=torch.float32, device=device)
        val = torch.tensor(val, dtype=torch.float32, device=device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        N = obs.shape[0]
        idx = np.arange(N)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, N, self.mb):
                j = idx[start:start+self.mb]
                logits, v = self.net(obs[j])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act[j])
                ratio = (logp - logp_old[j]).exp()
                surr1 = ratio * adv[j]
                surr2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * adv[j]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((ret[j] - v.squeeze(-1))**2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + 0.5*value_loss - self.ent*entropy
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()


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

def greedy_action_from_qtable(state):
    # state 可能是 tuple 或字串；確保 q_table 的 key 型別一致
    if state in q_table and q_table[state]:
        return max(q_table[state], key=q_table[state].get)
    return random.choice(actions)

def choose_action(state):
    # 終止或未知狀態處理
    if state == TERMINAL_STATE or state == "unknown":
        return "move 0"

    # ---- 資料蒐集模式：固定用 Q-table 貪婪策略（不加探索）----
    if args.collect:
        return greedy_action_from_qtable(state)

    # ---- 你原本的訓練模式（epsilon-greedy）----
    if random.random() < epsilon or state not in q_table:
        return random.choice(actions)
    else:
        # 用 get 避免 key error；若該 state 剛建立也能正常取最大值
        return max(q_table[state], key=q_table[state].get)


def update_q_table(state, action, reward, next_state):
    # 🔹 在資料蒐集模式（--collect）下不更新 Q-table
    #    這樣就只會用舊的 Q 值走路，不會覆蓋訓練成果
    if args.collect:
        return

    # 🔹 終止狀態或未知狀態不更新
    if state in [TERMINAL_STATE, "unknown"]:
        return

    # 🔹 若是新狀態就初始化動作字典
    if state not in q_table:
        q_table[state] = {a: 0.0 for a in actions}

    if next_state != TERMINAL_STATE and next_state != "unknown" and next_state not in q_table:
        q_table[next_state] = {a: 0.0 for a in actions}

    old_q_value = q_table[state].get(action, 0.0)

    # 🔹 計算下一狀態的最大 Q 值
    if next_state == TERMINAL_STATE or next_state == "unknown" or not q_table.get(next_state):
        next_max_q = 0.0
    else:
        if q_table[next_state]:  # 確保 next_state 存在且有動作
            next_max_q = max(q_table[next_state].values())
        else:
            next_max_q = 0.0

    # 🔹 Q-learning 更新公式
    q_table[state][action] = old_q_value + alpha * (reward + gamma * next_max_q - old_q_value)


# 主訓練流程
num_episodes = 500
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

    if args.ppo:
        obs_dim = 25
        n_act = len(actions)
        ppo = PPOAgent(obs_dim, n_act)
        rollout = RolloutBuf()

    step_id = 0  # ← 這裡加：每個 episode 開始先把步數歸零
    
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

        # === 選動作（可切換 PPO / 既有 Q-learning）===
        # 先給預設，避免後面變數未定義
        logp_t, val_t = 0.0, 0.0
        
        if args.ppo:
            # state 應該是 25 維浮點向量（用 --use_vision 時）
            if isinstance(state, (list, tuple)) and len(state) == 25:
                obs_vec = np.array(state, np.float32)
                a_idx, logp_t, val_t = ppo.select_action(obs_vec)   # ← 來自我們加的 PPOAgent
                action = actions[a_idx] 
            else:
                # 抓不到有效 state 就先隨機，以免當掉
                action = random.choice(actions)
        else:
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


        if args.collect and (step_id % args.save_every == 0):
            from PIL import Image
            import numpy as np, json, os
            from datetime import datetime

            os.makedirs(args.out, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
            base = os.path.join(args.out, f"ep{episode:04d}_step{step_id:05d}_{ts}")

            # --- 儲存影像 ---
            if world_state_after_action.is_mission_running and len(world_state_after_action.video_frames) > 0:
                frame = world_state_after_action.video_frames[-1]
                w, h = frame.width, frame.height
                img = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(h, w, 3)
                Image.fromarray(img).save(base + ".png")

            # --- 儲存 state / next_state / action / reward ---
            try:
                np.save(base + "_state.npy", np.array(state if isinstance(state, tuple) else ["unknown"] * 25, dtype=object))
                np.save(base + "_next.npy",  np.array(next_state if isinstance(next_state, tuple) else ["unknown"] * 25, dtype=object))
            except Exception as e:
                pass

            with open(base + "_action.txt", "w", encoding="utf-8") as f:
                f.write(action)
    
            with open(base + "_reward.json", "w", encoding="utf-8") as f:
                json.dump({
                    "env_reward": reward_from_env,
                    "stay_on_coal": current_stay_on_coal_reward,
                    "revisit_penalty": current_revisit_penalty,
                    "no_progress_penalty": current_no_progress_penalty,
                    "final_step_reward": float(reward_from_env)
                }, f, ensure_ascii=False)
        step_id += 1


        if action == "move 1" and block_in_front_center_prev_step == "coal_block":
            current_stay_on_coal_reward = STAY_ON_COAL_REWARD

        if next_state != TERMINAL_STATE and next_state != "unknown":
            if next_state in visited_states_in_episode:
                current_revisit_penalty = VISITED_PENALTY
            else:
                visited_states_in_episode.add(next_state)

            if state == next_state and state not in [TERMINAL_STATE, "unknown"]:
                current_no_progress_penalty = NO_PROGRESS_PENALTY

        final_reward_for_step = reward_from_env + current_revisit_penalty + STEP_PENALTY + current_stay_on_coal_reward + current_no_progress_penalty
        total_episode_reward += final_reward_for_step

        if args.ppo:
            done_flag = 0.0 if world_state_after_action.is_mission_running else 1.0
            if isinstance(state, (list, tuple)) and len(state) == 25:
                   obs_vec = np.array(state, dtype=np.float32)
                   rollout.add(
                       obs_vec,
                       actions.index(action),
                       float(final_reward_for_step),
                       float(logp_t),
                       float(val_t),
                       float(done_flag)
                   )
            # 每 2048 步更新一次，或任務結束時更新
            if len(rollout.obs) >= 2048 or done_flag == 1.0:
                buf = rollout.flush()
                ppo.update(buf)

        # === 🧩 若非 PPO / collect 模式，照舊更新 Q-table ===
        if (not args.ppo) and (not args.collect):
            update_q_table(state, action, final_reward_for_step, next_state)
    
        state = next_state
        current_world_state = world_state_after_action

        if not current_world_state.is_mission_running:
            break

    reason = "任務進行中"
    if not current_world_state.is_mission_running:
        if current_world_state.errors:
            reason = "❌ 錯誤導致結束"
            for err in current_world_state.errors: print("Mission Error:", err.text)
        else:
            is_timeout = any("description='time_up'" in msg.text for msg in current_world_state.mission_control_messages)
            if is_timeout:
                reason = "⏰ 超時"
            else:
                is_diamond_end = False
                for r_obj in world_state_after_action.rewards:
                    if abs(r_obj.getValue() - DIAMOND_REWARD_FROM_XML) < 1e-5:
                        is_diamond_end = True
                        break
                if is_diamond_end:
                    reason = f"💎 到達終點! (XML獎勵: {DIAMOND_REWARD_FROM_XML})"
                else:
                    reason = f"其他原因結束 (最後環境獎勵: {reward_from_env:.1f}, 可能踩到草地)"


    print(f"🎯 Episode {episode+1} 結束，總 reward: {total_episode_reward:.2f}, Steps: {step_count}, 原因: {reason}, Q-Table Size: {len(q_table)}")
    reward_log.append(total_episode_reward)
    timesteps_log.append(step_count)

    if not args.collect:
        if epsilon > min_epsilon:
         epsilon *= epsilon_decay_rate
         epsilon = max(min_epsilon, epsilon)

        time.sleep(0.5)

if args.collect:
    print("🟢 資料蒐集完成，不繪圖也不保存 Q-table。")
    exit(0)

# --- 訓練完成後保存 Q-table ---
if SAVE_Q_TABLE_ON_FINISH_ONLY or not Q_TABLE_FILENAME: # 如果 Q_TABLE_FILENAME 為 None，也在此生成文件名並保存
    save_filename = Q_TABLE_FILENAME
    if not save_filename: # 如果初始 Q_TABLE_FILENAME 為 None，則生成一個帶時間戳的新文件名
        save_filename = f"q_table_ep{num_episodes}_{int(time.time())}.json"
    
    print(f"\n💾 正在保存最終 Q-table 到 {save_filename}...")
    q_table_to_save = {}
    for state_tuple, actions_dict in q_table.items():
        if isinstance(state_tuple, str):
            str_state_key = state_tuple
        else:
            str_state_key = str(state_tuple)
        q_table_to_save[str_state_key] = actions_dict

    try:
        with open(save_filename, 'w') as f:
            json.dump(q_table_to_save, f, indent=4)
        print(f"✅ Q-table 已成功保存到 {save_filename}")
    except Exception as e:
        print(f"❌ 保存 Q-table 失敗: {e}")


print("\n🎉 訓練完成！")

# 繪圖部分 (保持不變)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
window_size = max(1, min(num_episodes // 10, 100 if num_episodes > 100 else num_episodes //10 ) )

if len(reward_log) >= window_size:
    moving_avg_rewards = np.convolve(reward_log, np.ones(window_size)/window_size, mode='valid')
    plt.plot(reward_log, label="Total reward per episode", alpha=0.5)
    plt.plot(np.arange(window_size - 1, len(reward_log)), moving_avg_rewards, label=f"{window_size}-episode Moving Avg", linewidth=2, color='red')
else:
    plt.plot(reward_log, label="Total reward per episode")

plt.title("Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
if len(timesteps_log) >= window_size:
    moving_avg_steps = np.convolve(timesteps_log, np.ones(window_size)/window_size, mode='valid')
    plt.plot(timesteps_log, label="Steps per episode", alpha=0.5)
    plt.plot(np.arange(window_size-1, len(timesteps_log)), moving_avg_steps, label=f"{window_size}-episode Moving Avg", linewidth=2, color='orange')
else:
    plt.plot(timesteps_log, label="Steps per episode")
plt.title("Steps per Episode")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()