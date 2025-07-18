import numpy as np
import time
import cv2

class Canvas:
    def __init__(self, frame_shape):
        """Initialize a blank canvas and stacks for undo/redo."""
        self.height, self.width, _ = frame_shape
        self._create_blank()
        self.actions = []
        self.redo_stack = []
        self.current_stroke = []

    def _create_blank(self):
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def start_stroke(self):
        """Begin a new stroke action."""
        self.current_stroke = []

    def add_segment(self, start_point, end_point, color, thickness):
        """Draw a segment on the canvas and record it."""
        # Draw antialiased line segment for all colors
        cv2.line(self.canvas, start_point, end_point, color, thickness, lineType=cv2.LINE_AA)
        # Draw circles at both endpoints to fill any gaps
        radius = max(1, thickness // 2)
        cv2.circle(self.canvas, start_point, radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(self.canvas, end_point, radius, color, -1, lineType=cv2.LINE_AA)
        # Record the segment
        self.current_stroke.append((start_point, end_point, color, thickness))

    def end_stroke(self):
        """End the current stroke and push it onto the action stack."""
        if self.current_stroke:
            self.actions.append(self.current_stroke)
            self.redo_stack.clear()
            self.current_stroke = []

    def undo(self):
        """Undo the last stroke action."""
        if self.actions:
            last = self.actions.pop()
            self.redo_stack.append(last)
            self._redraw()

    def redo(self):
        """Redo the most recently undone action."""
        if self.redo_stack:
            stroke = self.redo_stack.pop()
            self.actions.append(stroke)
            for seg in stroke:
                cv2.line(self.canvas, seg[0], seg[1], seg[2], seg[3])

    def clear(self):
        """Clear the canvas and reset history."""
        self.actions.clear()
        self.redo_stack.clear()
        self._create_blank()

    def save(self, filename=None):
        """Save the current canvas to a PNG file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"drawing_{timestamp}.png"
        cv2.imwrite(filename, self.canvas)
        print(f"Canvas saved as {filename}")
        return filename

    def get_canvas(self):
        """Get a copy of the current canvas image."""
        return self.canvas.copy()

    def flood_fill(self, seed_point, new_color):
        """Perform a flood-fill on the canvas at seed_point with new_color."""
        # Create mask with 2 pixels padding as required by OpenCV
        mask = np.zeros((self.height + 2, self.width + 2), np.uint8)
        # Tolerance for color difference
        lo = (10, 10, 10)
        hi = (10, 10, 10)
        cv2.floodFill(self.canvas, mask, seed_point, new_color, lo, hi, flags=cv2.FLOODFILL_FIXED_RANGE)

    def add_text(self, position, text, color, thickness):
        """Draw text onto the canvas at the given position."""
        # Scale text size relative to thickness
        font_scale = max(1, thickness / 10)
        cv2.putText(self.canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    def _redraw(self):
        """Redraw the canvas from the action history."""
        # Clear canvas
        self._create_blank()
        # Draw each stroke as a smooth antialiased polyline
        for stroke in self.actions:
            if not stroke:
                continue
            # Build sequence of points: start of first segment, then all end points
            pts = [stroke[0][0]] + [seg[1] for seg in stroke]
            pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            color = stroke[0][2]
            thickness = stroke[0][3]
            # Draw an antialiased polyline for smooth strokes
            cv2.polylines(self.canvas, [pts_np], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA) 