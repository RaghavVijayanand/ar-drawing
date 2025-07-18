import cv2

# Height in pixels for the button area.
button_height = 60

def draw_buttons(frame, labels):
    """Draw interactive buttons on the top of the frame and return the button width."""
    h, w, _ = frame.shape
    num = len(labels)
    button_width = w // num
    for i, label in enumerate(labels):
        x1 = i * button_width
        y1 = 0
        x2 = (i + 1) * button_width
        y2 = button_height
        # Draw filled rectangle for button background.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), -1)
        # Draw border for the button.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 50), 2)
        # Center the label text in the button.
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = x1 + (button_width - text_size[0]) // 2
        text_y = y1 + (button_height + text_size[1]) // 2
        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return button_width

def highlight_button(frame, index, labels, color=(0, 255, 0)):
    """Highlight a button border when hovered or active."""
    h, w, _ = frame.shape
    num = len(labels)
    button_width = w // num
    x1 = index * button_width
    y1 = 0
    x2 = (index + 1) * button_width
    y2 = button_height
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

def draw_thickness_slider(frame, x, y, length, thickness, value):
    """Draw a horizontal thickness slider with a knob at 'value' (0.0 to 1.0)."""
    # Slider track
    cv2.line(frame, (x, y), (x + length, y), (150, 150, 150), thickness)
    # Knob position
    knob_x = int(x + value * length)
    cv2.circle(frame, (knob_x, y), thickness * 2, (0, 0, 255), -1)
    return (x, y, length)

def draw_hud(frame, is_eraser, color, thickness):
    """Draw a heads-up display showing current mode, color, and thickness."""
    h, w, _ = frame.shape
    mode_text = 'Eraser' if is_eraser else 'Draw'
    color_text = f'Color: ({color[2]},{color[1]},{color[0]})'  # Show as RGB
    text = f'Mode: {mode_text} | {color_text} | Thickness: {thickness}'
    # Background rectangle
    cv2.rectangle(frame, (5, h - 30), (w - 5, h - 5), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def draw_help_overlay(frame):
    """Draw a semi-transparent help overlay with gesture and key instructions."""
    h, w, _ = frame.shape
    overlay = frame.copy()
    lines = [
        "Help - Gestures & Controls:",
        "Index Up: Draw", 
        "Pinch or Fist: Toggle Eraser", 
        "Two-Fingers Up: Shape Mode", 
        "Three-Fingers Up: Flood-Fill", 
        "Peace Sign: Text Annotation", 
        "h: Toggle Help Overlay", 
        "u/r: Undo/Redo", 
        "c: Clear", 
        "q: Quit"
    ]
    y = 50
    for line in lines:
        cv2.putText(overlay, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 30
    # Blend overlay
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame) 