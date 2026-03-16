import argparse
import datetime
import json
import logging
import os
import time
import urllib.error
import urllib.request

from experiment_runner import load_results, run_config
from search_space import search_space

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL")
model = os.getenv("LLM_MODEL")

DEFAULT_PROMPT_FILE = "prompts/llm_config_prompt.txt"
LLM_RUNS_FILE = "results/llm_runs.jsonl"
LOGGER = logging.getLogger(__name__)

SUPPORTED_KEYS = [
    "seed",
    "use_shooting",
    "use_free_throws",
    "use_rebounding",
    "use_turnovers",
    "use_defense",
    "use_playmaking",
    "use_win_history",
    "n_estimators",
    "learning_rate",
    "max_depth",
    "sample_size",
]


def ensure_dir(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def read_prompt(path):
    with open(path, "r") as f:
        return f.read().strip()


def format_search_space(space, keys):
    lines = []
    for key in keys:
        if key not in space:
            raise ValueError(f"Missing search space for key: {key}")
        lines.append(f"{key}: {space[key]}")
    return "\n".join(lines)


def build_user_prompt(history, space_text):
    history_payload = json.dumps(history, indent=2)
    return (
        "Allowed values:\n"
        f"{space_text}\n\n"
        "Previous results (ordered oldest to newest):\n"
        f"{history_payload}\n\n"
        "Return the next config JSON only."
    )


def call_chat_completion(messages, model, temperature, base_url, api_key):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": temperature}
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8")
        raise ValueError(
            f"HTTP {exc.code} from LLM API: {error_body}"
        ) from exc

    parsed = json.loads(body)
    return parsed["choices"][0]["message"]["content"]


def extract_json(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in response.")

    return json.loads(cleaned[start:end + 1])


def validate_config(config, keys, space):
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object.")

    extra = set(config) - set(keys)
    missing = set(keys) - set(config)
    if extra:
        raise ValueError(f"Config has extra keys: {sorted(extra)}")
    if missing:
        raise ValueError(f"Config is missing keys: {sorted(missing)}")

    for key in keys:
        if config[key] not in space[key]:
            raise ValueError(
                f"Invalid value for {key}: {config[key]} (allowed: {space[key]})"
            )

    return config


def get_next_config(
    base_prompt,
    user_prompt,
    model,
    temperature,
    base_url,
    api_key,
    max_retries,
):
    messages = [
        {"role": "system", "content": base_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_error = None
    for _ in range(max_retries):
        response = call_chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
        )
        try:
            config = extract_json(response)
            config = validate_config(config, SUPPORTED_KEYS, search_space)
            return config, response
        except ValueError as exc:
            last_error = str(exc)
            messages.append({"role": "assistant", "content": response})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That response was invalid: "
                        f"{last_error}. Return ONLY the JSON object."
                    ),
                }
            )

    raise ValueError(f"Failed to get valid config: {last_error}")


def save_llm_run(system_prompt, user_prompt, response, record):
    ensure_dir(LLM_RUNS_FILE)
    run = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response": response,
        "config": record["config"],
        "log_loss": record["log_loss"],
    }
    with open(LLM_RUNS_FILE, "a") as f:
        f.write(json.dumps(run) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0)
    parser.add_argument("--model", default=model)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--base-url", default=base_url)
    parser.add_argument("--api-key", default=api_key)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.api_key:
        raise ValueError("LLM_API_KEY is required.")
    if not args.base_url:
        raise ValueError("LLM_BASE_URL is required.")
    if not args.model:
        raise ValueError("LLM_MODEL is required.")

    base_prompt = read_prompt(args.prompt_file)
    space_text = format_search_space(search_space, SUPPORTED_KEYS)

    iteration = 0
    while True:
        LOGGER.info("Iteration %s: preparing prompt.", iteration + 1)
        history = load_results()
        user_prompt = build_user_prompt(history, space_text)

        LOGGER.info("Iteration %s: requesting next config.", iteration + 1)
        config, response = get_next_config(
            base_prompt=base_prompt,
            user_prompt=user_prompt,
            model=args.model,
            temperature=args.temperature,
            base_url=args.base_url,
            api_key=args.api_key,
            max_retries=args.max_retries,
        )
        config.update(FIXED_CONFIG)

        LOGGER.info("Iteration %s: running experiment.", iteration + 1)
        record = run_config(config)
        LOGGER.info("Iteration %s: log loss %s.", iteration + 1, record["log_loss"])
        save_llm_run(base_prompt, user_prompt, response, record)
        LOGGER.info("Iteration %s: saved run details.", iteration + 1)

        iteration += 1
        if args.max_iterations is not None and iteration >= args.max_iterations:
            LOGGER.info("Reached max iterations (%s).", args.max_iterations)
            break
        if args.sleep_seconds > 0:
            LOGGER.info("Sleeping for %s seconds.", args.sleep_seconds)
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
