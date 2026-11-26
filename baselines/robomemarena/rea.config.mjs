import { fileURLToPath } from "node:url";
import { createPythonDriverEnvironment } from "../REA/src/environments/python-driver.mjs";
import { createChatPolicy } from "../REA/src/framework/index.mjs";

const project = fileURLToPath(new URL("../..", import.meta.url));
const directory = fileURLToPath(new URL(".", import.meta.url));
const task = Number(process.env.REA_RMA_TASK ?? 1);
const seed = Number(process.env.REA_RMA_SEED ?? 50);

export default {
  name: "RoboMemArena",
  episodes: [{ id: `task${task}-seed${seed}`, task, seed }],
  budget: { maxSteps: 2500, maxDurationMs: 1800000, decisionTimeoutMs: 120000 },
  createEnvironment() {
    return createPythonDriverEnvironment({
      id: "robomemarena", execute: true, requestTimeoutMs: 120000,
      python: `${project}/.venv/bin/python`, driver: "rea_driver:RoboMemArenaDriver",
      env: { ...process.env, PYTHONPATH: `${directory}:${project}` },
      options: {
        envPort: process.env.BASELINE_ENV_PORT,
        policyPort: process.env.BASELINE_VLA_PORT,
        maxSteps: Number(process.env.REA_RMA_MAX_PHYSICAL_STEPS ?? 2500),
      },
    });
  },
  createPolicy() {
    if (process.env.REA_RMA_POLICY === "chat") {
      return createChatPolicy({
        model: process.env.REA_MODEL, baseUrl: process.env.REA_BASE_URL, apiKey: process.env.REA_API_KEY,
        mode: process.env.REA_MODEL === process.env.BASELINE_QWEN_MODEL ? "json" : "tool",
      });
    }
    return {
      id: "pi05:robomemarena",
      async decide({ observation }) {
        return { action: { name: "vla_chunk", arguments: { prompt: observation.text } } };
      },
    };
  },
};
