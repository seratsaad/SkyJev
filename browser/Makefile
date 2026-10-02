CHROME=/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome
PROFILE=$(CURDIR)/.chrome-profile

chrome:  ## start a dedicated Chrome with remote debugging on :9222 (scratch profile)
	@mkdir -p $(PROFILE)
	@curl -s http://127.0.0.1:9222/json/version >/dev/null 2>&1 && echo "chrome already listening on 9222" || \
	 (nohup $(CHROME) --remote-debugging-port=9222 --user-data-dir=$(PROFILE) --no-first-run --no-default-browser-check --window-size=1200,900 about:blank >$(PROFILE)/chrome.log 2>&1 & sleep 3; curl -s http://127.0.0.1:9222/json/version | head -2)

validate:  ## check every verifier passes on its gold page and fails on its start page
	uv run --env-file .env python -m astroweb.runner validate

credit:  ## check the model account can actually pay for a call
	@uv run --env-file .env python -c "import os,httpx;r=httpx.post('https://api.openai.com/v1/chat/completions',headers={'Authorization':'Bearer '+os.environ['OPENAI_API_KEY']},json={'model':'gpt-4.1-mini','max_tokens':5,'messages':[{'role':'user','content':'say ok'}]},timeout=30);print('OK:',r.json()['choices'][0]['message']['content']) if r.status_code==200 else print('NOT OK:',r.status_code,r.text[:200])"

jev:  ## the JEV arm, 3 repeats (needs TYPESAFE_API_KEY; the chooser is small, so it is cheap)
	uv run --env-file .env python -m astroweb.runner run --agent jev --repeat 3

baseline:  ## the Browser Use arm, 2 repeats plus the free-mode lookup comparison, inside the ceiling
	uv run --env-file .env python -m astroweb.runner run --agent baseline --repeat 2
	uv run --env-file .env python -m astroweb.runner run --agent baseline --mode free --repeat 1 --category lookup

gpt-jev:  ## JEV's action space with gpt-4.1-mini as the chooser, 3 repeats (needs OPENAI_API_KEY)
	uv run --env-file .env python -m astroweb.runner run --agent gpt-jev --repeat 3

summary:  ## rebuild results/summary/ from the raw runs under results/<arm>/
	uv run python analysis/summarize_results.py --print

survey:  ## what the agent's page snapshot sees on the science sites, no model calls (results/site_survey.csv)
	uv run --env-file .env python analysis/site_survey.py

ship:  ## compact the saved runs so the repository can carry them
	uv run python analysis/compact_results.py

test:
	uv run pytest -q
