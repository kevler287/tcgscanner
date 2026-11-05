import cv2
import numpy as np

from client.common.detectionstate_enum import DetectionState


PANEL_WIDTH = 900
BAR_HEIGHT = 60
BAR_PADDING = 30
FONT = cv2.FONT_HERSHEY_SIMPLEX

def build_progress_panel(progress: dict, status: DetectionState, product: dict, panel_height: int) -> np.ndarray:
    if status in [DetectionState.IDENTIFIED, DetectionState.AMBIGUOUS, DetectionState.ERRONEOUS]:
        panel = np.full((panel_height, PANEL_WIDTH, 3), 30, dtype=np.uint8)

        center_y = panel_height // 2

        if status == DetectionState.IDENTIFIED:
            edition = "1st Edition" if product["x-isfirst"] else "Unlimited"
            lines = [product["x-setcode"], edition]
        else:
            lines = [status.name]

        font_scale = 2
        thickness = 3
        line_height = cv2.getTextSize("A", FONT, font_scale, thickness)[0][1] + 15  # + line spacing

        for i, line in enumerate(lines):
            text_size = cv2.getTextSize(line, FONT, font_scale, thickness)[0]
            text_x = (PANEL_WIDTH - text_size[0]) // 2
            text_y = center_y + 20 + i * line_height
            cv2.putText(panel, line, (text_x, text_y), FONT, font_scale, status.get_color_gbr(), thickness, cv2.LINE_AA)

        return panel
    elif status == DetectionState.RUNNING:
        panel = np.full((panel_height, PANEL_WIDTH, 3), 30, dtype=np.uint8)

        y = BAR_PADDING
        for field, (value, fraction) in progress.items():
            label = f"{field}: {value}"
            cv2.putText(panel, label, (10, y + 15), FONT, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

            bar_y = y + 25
            cv2.rectangle(panel, (10, bar_y), (PANEL_WIDTH - 10, bar_y + 10), (80, 80, 80), -1)

            fill_width = int((PANEL_WIDTH - 20) * min(max(abs(fraction), 0), 1))
            color = (0, 200, 0) if abs(fraction) >= 1.0 else (0, 165, 255)
            cv2.rectangle(panel, (10, bar_y), (10 + fill_width, bar_y + 10), color, -1)

            y += BAR_HEIGHT + BAR_PADDING

        return panel
    return None

def build_live_ui(frame: np.ndarray, progress: dict, status: DetectionState, product: dict) -> np.ndarray:
    progress_panel = build_progress_panel(progress=progress, status=status, product=product, panel_height=frame.shape[0])
    # match panel height to frame height
    if progress_panel.shape[0] != frame.shape[0]:
        resized_panel = np.full((frame.shape[0], PANEL_WIDTH, 3), 30, dtype=np.uint8)
        resized_panel[:progress_panel.shape[0], :] = progress_panel
        progress_panel = resized_panel

    return np.hstack([frame, progress_panel])