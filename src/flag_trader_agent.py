import torch
import random
import re
import json
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt_generator import build_prompt

# src/flag_trader_agent.py
class FlagTraderAgent(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device("mps" if cfg["device"] == "mps" and torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"], torch_dtype=torch.float16
        ).to(self.device)
        n_total = len(self.model.model.layers)
        for i, layer in enumerate(self.model.model.layers):
            layer.requires_grad_(i >= n_total - cfg["freeze_layers"])
        lr = float(cfg["learning_rate"])  # 문자열을 float로 변환
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )
        self.cfg = cfg
        self.action_map = {"Buy": 0, "Sell": 1, "Hold": 2}
        
    def _infer_action(self, obs_dict) -> int:
        prompt = build_prompt(obs_dict)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=8)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        m = re.search(r"\{.*?\}", text)
        if m:
            try:
                action_json = json.loads(m.group(0).replace("'", '"'))
                return self.action_map.get(action_json.get("Action", "Hold"), 2)
            except json.JSONDecodeError:
                pass
        return 2  # Default Hold

    def train_one_episode(self, env):
        obs, _ = env.reset()
        traj = []
        log_probs = []
        for step in range(self.cfg["warmup_steps"] + self.cfg["test_steps"]):
            obs_dict = {
                "price": obs[0], "volume": obs[1], "rsi": obs[2],
                "cash": obs[3], "shares": obs[4]
            }
            action = self._infer_action(obs_dict)
            next_obs, reward, done, _, _ = env.step(action)
            traj.append((obs, action, reward))
            if step < self.cfg["warmup_steps"]:
                prompt = build_prompt(obs_dict)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    log_prob = torch.log_softmax(logits[:, -1, :], dim=-1).mean()
                log_probs.append(log_prob)
            obs = next_obs
            if done:
                break
        if log_probs:
            advantage = torch.tensor([r for _, _, r in traj[:len(log_probs)]], device=self.device).mean()
            loss = (-advantage.detach() * torch.stack(log_probs)).mean()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return traj
