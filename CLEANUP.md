# Cleaning up

None of this is required. If you plan to keep experimenting, keep everything — the models are the
slow part to get back.

But this workshop pulls a multi-gigabyte model onto the machine and possibly installs Ollama. If
you want the machine back as it was — a shared laptop, a training room, a CI box — here is every
piece of it, in the order worth doing. The models are almost all of the disk space.

| What | Where it lives | Size |
|---|---|---|
| `agentfix-mellum2` + its base model | Ollama's model store | ~8 GB |
| `agentfix-qwen3` + `qwen3:1.7b` | Ollama's model store | ~1.4 GB |
| Ollama itself | see below, per OS | a few hundred MB |
| `agentfix-sandbox` Docker image | Docker, only if you ran the sandbox demo | a few hundred MB |
| `.venv`, `uv` caches | inside the clone, and `~/.cache/uv` | a few hundred MB |
| `MELLUM_MODEL` | your shell session only, unless you persisted it | — |

## 1. Remove the models

Identical on macOS, Linux, WSL2 and Windows, and it reclaims almost all of the disk space.

```bash
ollama list
```

Then remove what this workshop created — you only have the pair for the option you used:

```bash
ollama rm agentfix-mellum2 hf.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q4_K_M
```

```bash
ollama rm agentfix-qwen3 qwen3:1.7b
```

Remove the derived `agentfix-…` model **and** the base model it was built from. Deleting only the
derived one leaves the multi-gigabyte download in place, which is the part actually using the disk;
deleting only the base leaves a derived model pointing at nothing.

Anything you pulled before this workshop is untouched.

<details>
<summary><b>Where the files were, if the space did not come back</b></summary>

- macOS, and Linux when installed as your own user: `~/.ollama/models`
- Linux with the systemd service: `/usr/share/ollama/.ollama/models`
- Windows: `%USERPROFILE%\.ollama\models`

If `ollama list` is empty but that directory is still large, layers are shared with a model you
kept, or the server was running mid-delete. Restart Ollama and look again before deleting anything
there by hand.
</details>

## 2. The `MELLUM_MODEL` variable

In this repo the variable is normally set per command or per shell session, so closing the terminal
is the whole cleanup:

```bash
unset MELLUM_MODEL                 # macOS, Linux, WSL2
Remove-Item Env:\MELLUM_MODEL      # PowerShell
```

If you made it permanent yourself, undo it the same way you set it: remove the `export` line from
`~/.zshrc` / `~/.bashrc` / `~/.config/fish/config.fish`, or on Windows run

```powershell
reg delete HKCU\Environment /F /V MELLUM_MODEL
```

Use `reg delete` rather than `setx MELLUM_MODEL ""` — `setx` with an empty value leaves the
variable present but empty, which reads as a model named `""` and fails more confusingly than a
missing variable does. Already-running programs keep the old value until they restart.

While you are here, delete `Modelfile.agentfix-qwen3` if the Windows path of Option 2 created it
in the repo root. Do **not** delete `Modelfile` — that one ships with the repo.

## 3. Remove Ollama itself

Only if you installed it for this workshop. Remove the models first: uninstalling does not reliably
take the model store with it, and an 8 GB directory left behind is the thing you were trying to
reclaim.

<details>
<summary><b>macOS</b></summary>

The app: quit Ollama from the menu bar, then drag **Ollama.app** from **Applications** to the
Trash.

Homebrew:

```bash
brew services stop ollama
brew uninstall ollama
```

Then the leftovers, which neither route removes:

```bash
rm -rf ~/.ollama
sudo rm -f /usr/local/bin/ollama
```
</details>

<details>
<summary><b>Linux and WSL2</b></summary>

The install script registers a systemd service, so stop and remove that before the binary:

```bash
sudo systemctl stop ollama
sudo systemctl disable ollama
sudo rm -f /etc/systemd/system/ollama.service
sudo systemctl daemon-reload

sudo rm -f "$(command -v ollama)"
rm -rf ~/.ollama
```

The installer also creates a dedicated `ollama` user and its own model store. If nothing else on
the machine uses them:

```bash
sudo rm -rf /usr/share/ollama
sudo userdel ollama
sudo groupdel ollama
```

On WSL2 without systemd you started the server with `ollama serve`, so there is no service to
remove — stop that process, then delete the binary and `~/.ollama`.
</details>

<details>
<summary><b>Windows</b></summary>

**Settings** → **Apps** → **Installed apps** → **Ollama** → **Uninstall**, or from a terminal if
you installed it with winget:

```powershell
winget uninstall -e --id Ollama.Ollama
```

Then the directories the uninstaller leaves behind:

```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\.ollama"
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\Ollama"
```
</details>

## 4. The rest

<details>
<summary><b>The Docker sandbox image</b></summary>

Only if you built it:

```bash
docker rmi agentfix-sandbox
```

`docker image ls` confirms it is gone. If Docker says the image is in use, clear the stopped
container first with `docker container prune`.
</details>

<details>
<summary><b>The clone, its environment, and uv's caches</b></summary>

The virtual environment lives inside the clone, so deleting the clone takes the packages with it —
nothing was installed into your system Python.

```bash
rm -rf .venv                       # just the environment; uv sync rebuilds it
uv cache clean                     # uv's download cache, shared across projects
```

`uv` also keeps any interpreters it downloaded. `uv python list` shows them and
`uv python uninstall 3.12` removes one. To remove uv itself:

```bash
rm -rf ~/.local/share/uv ~/.local/bin/uv ~/.local/bin/uvx
```

A Python that came from `brew`, `apt`, `dnf`, `pacman` or `winget` is an ordinary package —
uninstall it with the same tool, or keep it.
</details>

<details>
<summary><b>Colab (Option 3)</b></summary>

Nothing to clean. The Colab runtime is discarded when the session ends, and the model, the Ollama
install and the cloned repository go with it. Your own machine never had them.

Delete the notebook copy from your Drive if Colab saved one there.
</details>

## Check it worked

```bash
ollama list          # no agentfix-… entries, or "command not found" if Ollama is gone
echo $MELLUM_MODEL   # empty
```

`agentfix doctor` will now fail, which is the point — you removed what it checks for. Every
exercise test still passes, because they run against a scripted fake model and never needed any of
this.
