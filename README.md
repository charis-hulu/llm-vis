# Camera Reasoning App

A notebook-first, manual LLM-guided VTK camera alignment system.

## Quick Start

```bash
pip install -r requirements.txt
```

Place your raw volume at `data/foot_256x256x256_uint8.raw`, then open the notebook:

```bash
jupyter notebook notebooks/manual_chatgpt_loop.ipynb
```

Or run the script equivalent:

```bash
python examples/manual_chatgpt_loop_example.py
```

## Workflow

1. Run the **Initialize** cell (or script). This loads the volume, renders it, saves a screenshot to `output/screenshots/latest.png`, and writes the LLM prompt to `output/llm_prompt.txt`.
2. Copy the prompt text and the screenshot and send them to ChatGPT.
3. Paste ChatGPT's full response into the **Process Response** cell and run it. The notebook will:
   - Extract the chosen action.
   - Apply it to the VTK camera.
   - Save a new screenshot (e.g. `step_001_ELEVATION_UP_MEDIUM.png`).
   - Save the camera state JSON.
   - Update `action_history.json`.
   - Write the next LLM prompt.
4. Repeat until ChatGPT returns `STOP`.

## Project Structure

```
camera_reasoning_app/
  requirements.txt
  camera_reasoning/
    __init__.py
    session.py          # CameraReasoningSession — main API
    camera_actions.py   # Action definitions and apply_action()
    camera_state.py     # get/set/save/load camera state helpers
    volume_scene.py     # VTK scene construction and screenshot saving
    prompt_writer.py    # LLM prompt generation
    action_parser.py    # ChatGPT response parsing
  notebooks/
    manual_chatgpt_loop.ipynb
  examples/
    manual_chatgpt_loop_example.py
  output/
    screenshots/
    camera_states/
    action_history.json
    llm_prompt.txt
  data/
    foot_256x256x256_uint8.raw   # place your volume here
```

## Extending Later

| Goal | What to add |
|---|---|
| GUI | Wrap `CameraReasoningSession` in a Qt/Tk window; wire buttons to `process_chatgpt_response` |
| File watcher | Watch `output/chatgpt_response.txt` for changes and auto-call `process_chatgpt_response` |
| Full autonomous agent | Replace the manual paste step with an API call in `session.py` |
| Mesh support | Add `build_mesh_pipeline(path)` to `volume_scene.py` |
| Volume rendering | Replace `build_isosurface_pipeline` with `vtkSmartVolumeMapper` pipeline |
