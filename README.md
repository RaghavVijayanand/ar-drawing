# Virtual Hand-Tracking Drawing App with AR

This interactive Python application lets you draw on a live webcam feed with hand gestures and manipulate real-world objects with AR capabilities. It uses MediaPipe for hand detection, OpenCV for rendering, and PyTorch with MiDaS for depth estimation.

## Features

- **Real-time hand-tracking** (MediaPipe) with background processing thread for high FPS
- **Drawing**: index finger up alone, with smoothing and antialiased strokes
- **Eraser**: pinch or fist gesture toggles eraser mode
- **Shape drawing**: two-finger tap marks start/end to draw rectangles or circles
- **Flood-fill**: three-finger tap fills an area with the selected color
- **Text annotations**: peace sign gesture brings up an on-screen prompt to type text
- **Interactive UI**: color buttons, undo/redo, clear/save, thickness slider at bottom
- **Heads-up display** (HUD) and **Help overlay** with gesture & key cheat-sheet
- **Logging**: events, gestures, and key actions are logged for debugging

### AR Features

- **Object selection and manipulation**: use pinch gesture to pick up and move real-world objects
- **Edge detection**: automatically identifies object boundaries for seamless selection
- **Depth estimation**: powered by MiDaS deep learning model for monocular depth perception
- **Dynamic scaling**: objects scale based on relative depth when moved
- **Automatic inpainting**: fills in the background when objects are moved with OpenCV inpainting

## Installation & Requirements

1. Clone or download this repository.
2. Install Python ≥ 3.7.
3. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\\Scripts\\activate  # Windows PowerShell
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the App

```bash
python main.py
```

- Press **q** to quit
- Press **c** to clear canvas
- Press **u** / **r** for undo/redo
- Press **h** to toggle help overlay
- Press **a** to toggle AR mode

## Gesture Cheat-Sheet

| Gesture             | Action                     |
|---------------------|----------------------------|
| Index finger up     | Draw (smoothing enabled)   |
| Pinch or fist       | Toggle eraser mode / Pick up object (in AR mode) |
| Two fingers up      | Shape mode (tap start/end) |
| Three fingers up    | Flood-fill at pointer      |
| Peace sign          | Text annotation prompt     |
| Fist                | Reset scene (in AR mode)   |

## AR Mode Controls

- Click the "AR Mode" button or press **a** to toggle AR mode
- In AR mode:
  - Use **pinch** gesture on an object to pick it up
  - Move your hand to position the object
  - Release pinch to place the object
  - Make a **fist** to reset the scene
  - The depth map visualization appears in the top-left corner

## Troubleshooting & Tips

- Ensure your webcam is connected and accessible (default index 0).
- If FPS is low, close other applications or lower camera resolution in `main.py`.
- For false detections, adjust `min_detection_confidence` in `gesture_detection.py`.
- If depth estimation fails, ensure PyTorch is properly installed with CUDA support for better performance.
- For optimal AR experience, use a well-lit environment with distinct objects.

---
*Developed with OpenCV, MediaPipe, MiDaS, PyTorch, and Python.* 