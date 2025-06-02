import cv2
import mediapipe as mp
import time
import pynput.keyboard

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

HOVER_DURATION_THRESHOLD = 0.5
CURSOR_BLINK_RATE = 0.5

keyboard_controller = pynput.keyboard.Controller()

current_hover_key = None
hover_start_time = None
last_pressed_key_char = None
typed_text = ""
is_shift_active = False

cursor_on = True
last_cursor_toggle_time = time.time()

KEY_HEIGHT_RATIO = 0.12
KEY_WIDTH_RATIO = 0.075
HORIZONTAL_PADDING_RATIO = 0.005
VERTICAL_PADDING_RATIO = 0.015

TEXT_AREA_HEIGHT_RATIO = 0.25

KEY_LAYOUT = [
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '='],
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    ['SHIFT', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', '<-'],
    ['ENTER', 'SPACE']
]

SHIFT_MAP = {
    '1': '!', '2': '@', '3': '#', '4': '$', '5': '%', '6': '^', '7': '&', '8': '*', '9': '(', '0': ')',
    '-': '_', '=': '+',
}

keys_on_screen = []

class Key:
    def __init__(self, char, x, y, w, h):
        self.char = char
        self.x = x
        self.y = y
        self.width = w
        self.height = h
        self.is_hovered = False
        self.is_pressed = False
        self.last_press_time = 0

    def contains(self, px, py):
        return self.x < px < self.x + self.width and \
               self.y < py < self.y + self.height

    def draw(self, frame, is_active=False, is_shift_key_active_global=False, SHIFT_MAP_REF=None):
        fill_color = (200, 200, 200)
        if is_active:
            fill_color = (0, 255, 0)
        elif self.char == 'SHIFT' and is_shift_key_active_global:
            fill_color = (255, 200, 0)

        border_color = (0, 0, 0)
        text_color = (0, 0, 0)

        cv2.rectangle(frame, (self.x, self.y), (self.x + self.width, self.y + self.height), fill_color, cv2.FILLED)
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.width, self.y + self.height), border_color, 2)

        display_char = self.char
        if display_char not in ['SPACE', '<-', 'SHIFT', 'ENTER']:
            if 'A' <= display_char <= 'Z':
                display_char = display_char.upper() if is_shift_key_active_global else display_char.lower()
            elif SHIFT_MAP_REF and display_char in SHIFT_MAP_REF:
                display_char = SHIFT_MAP_REF[display_char] if is_shift_key_active_global else display_char

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_size = cv2.getTextSize(display_char, font, font_scale, font_thickness)[0]
        text_x = self.x + (self.width - text_size[0]) // 2
        text_y = self.y + (self.height + text_size[1]) // 2
        cv2.putText(frame, display_char, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)


def calculate_key_positions(frame_width, frame_height):
    global keys_on_screen
    keys_on_screen = []

    keyboard_area_start_y = 0
    keyboard_area_height = int(frame_height * (1 - TEXT_AREA_HEIGHT_RATIO))

    key_h = int(keyboard_area_height * KEY_HEIGHT_RATIO)
    key_w = int(frame_width * KEY_WIDTH_RATIO)

    horiz_padding = int(frame_width * HORIZONTAL_PADDING_RATIO)
    vert_padding = int(keyboard_area_height * VERTICAL_PADDING_RATIO)

    y_offset = keyboard_area_start_y + vert_padding

    for row_index, row in enumerate(KEY_LAYOUT):
        current_row_keys_width = 0
        for char in row:
            char_key_w = key_w
            if char == 'SPACE':
                char_key_w = int(key_w * 5)
            elif char == 'SHIFT' or char == '<-' or char == 'ENTER':
                char_key_w = int(key_w * 1.8)
            current_row_keys_width += char_key_w + horiz_padding

        current_row_keys_width -= horiz_padding
        if current_row_keys_width < 0:
             current_row_keys_width = 0

        x_offset = int((frame_width - current_row_keys_width) / 2)

        for col_index, char in enumerate(row):
            current_key_w = key_w
            if char == 'SPACE':
                current_key_w = int(key_w * 5)
            elif char == 'SHIFT' or char == '<-' or char == 'ENTER':
                current_key_w = int(key_w * 1.8)

            x = x_offset
            y = y_offset

            keys_on_screen.append(Key(char, x, y, current_key_w, key_h))
            x_offset += current_key_w + horiz_padding

        y_offset += key_h + vert_padding


def get_key_at_position(px, py):
    for key in keys_on_screen:
        if key.contains(px, py):
            return key
    return None

def write_multiline_text(img, text, org, font, fontScale, color, thickness, lineType, line_height_offset=30):
    y0, dy = org[1], line_height_offset
    lines = text.split('\n')

    for i, line in enumerate(lines):
        y = y0 + i * dy
        cv2.putText(img, line, (org[0], y), font, fontScale, color, thickness, lineType)


def run_virtual_keyboard():
    global current_hover_key, hover_start_time, last_pressed_key_char, typed_text, is_shift_active
    global cursor_on, last_cursor_toggle_time

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read initial frame for dimensioning.")
        cap.release()
        return

    frame_height, frame_width, _ = frame.shape
    print(f"Camera opened with resolution: {frame_width}x{frame_height}")

    cv2.namedWindow('Virtual Camera Keyboard', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Virtual Camera Keyboard', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    calculate_key_positions(frame_width, frame_height)

    print("Virtual Keyboard active. Press 'q' to quit.")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    single_space_width = cv2.getTextSize(" ", font, font_scale, font_thickness)[0][0]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera. Exiting.")
            break

        frame = cv2.flip(frame, 1)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        finger_tip_coords = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, c = frame.shape
                index_tip_x = int(index_finger_tip.x * w)
                index_tip_y = int(index_finger_tip.y * h)
                finger_tip_coords = (index_tip_x, index_tip_y)

                cv2.circle(frame, (index_tip_x, index_tip_y), 10, (255, 0, 0), cv2.FILLED)

        detected_key_this_frame = None
        if finger_tip_coords:
            detected_key_this_frame = get_key_at_position(finger_tip_coords[0], finger_tip_coords[1])

        for key in keys_on_screen:
            key.is_hovered = False

        if detected_key_this_frame:
            detected_key_this_frame.is_hovered = True

            if current_hover_key is None:
                current_hover_key = detected_key_this_frame
                hover_start_time = time.time()

            elif current_hover_key.char == detected_key_this_frame.char:
                if time.time() - hover_start_time >= HOVER_DURATION_THRESHOLD:
                    if last_pressed_key_char != current_hover_key.char:
                        current_hover_key.is_pressed = True

                        char_to_process = current_hover_key.char
                        final_char_to_send = None

                        if char_to_process == 'SPACE':
                            keyboard_controller.press(pynput.keyboard.Key.space)
                            keyboard_controller.release(pynput.keyboard.Key.space)
                            final_char_to_send = ' '
                        elif char_to_process == '<-':
                            keyboard_controller.press(pynput.keyboard.Key.backspace)
                            keyboard_controller.release(pynput.keyboard.Key.backspace)
                            typed_text = typed_text[:-1] if typed_text else ""
                        elif char_to_process == 'ENTER':
                            keyboard_controller.press(pynput.keyboard.Key.enter)
                            keyboard_controller.release(pynput.keyboard.Key.enter)
                            final_char_to_send = '\n'
                        elif char_to_process == 'SHIFT':
                            is_shift_active = not is_shift_active
                            final_char_to_send = None
                        else:
                            if is_shift_active:
                                if 'A' <= char_to_process <= 'Z':
                                    final_char_to_send = char_to_process.upper()
                                elif char_to_process in SHIFT_MAP:
                                    final_char_to_send = SHIFT_MAP[char_to_process]
                                else:
                                    final_char_to_send = char_to_process
                            else:
                                if 'A' <= char_to_process <= 'Z':
                                    final_char_to_send = char_to_process.lower()
                                else:
                                    final_char_to_send = char_to_process

                            if final_char_to_send:
                                keyboard_controller.press(final_char_to_send)
                                keyboard_controller.release(final_char_to_send)

                        if final_char_to_send is not None and char_to_process != '<-':
                               typed_text += final_char_to_send

                        last_pressed_key_char = current_hover_key.char
                        current_hover_key.last_press_time = time.time()
                        cursor_on = True
                        last_cursor_toggle_time = time.time()
            else:
                current_hover_key = detected_key_this_frame
                hover_start_time = time.time()
                last_pressed_key_char = None
        else:
            current_hover_key = None
            hover_start_time = None
            last_pressed_key_char = None

        for key in keys_on_screen:
            active_for_drawing = key.is_hovered or \
                                 (key.is_pressed and (time.time() - key.last_press_time < 0.2))
            key.draw(frame, active_for_drawing, is_shift_active, SHIFT_MAP)
            key.is_pressed = False

        text_area_start_y = int(frame_height * (1 - TEXT_AREA_HEIGHT_RATIO))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, text_area_start_y), (frame_width, frame_height), (50, 50, 50), -1)
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        text_color = (255, 255, 255)
        text_origin_x = 50
        text_origin_y = text_area_start_y + 40
        line_height_offset = 30

        display_lines = []
        current_line_buffer = ""
        avg_char_width = cv2.getTextSize("A", font, font_scale, font_thickness)[0][0]
        max_chars_per_line_est = int((frame_width - text_origin_x * 2) / avg_char_width)

        segments = typed_text.split('\n')
        for seg_idx, segment in enumerate(segments):
            temp_segment_buffer = ""
            for char_idx, char in enumerate(segment):
                if len(temp_segment_buffer) + 1 > max_chars_per_line_est:
                    last_space_idx = temp_segment_buffer.rfind(' ')
                    if last_space_idx > 0:
                        display_lines.append(temp_segment_buffer[:last_space_idx])
                        temp_segment_buffer = temp_segment_buffer[last_space_idx+1:]
                    else:
                        display_lines.append(temp_segment_buffer)
                        temp_segment_buffer = ""

                temp_segment_buffer += char

            if temp_segment_buffer:
                display_lines.append(temp_segment_buffer)

            if seg_idx < len(segments) - 1:
                display_lines.append("")

        if not display_lines:
            display_lines.append("")

        max_lines_to_show = int((frame_height - text_area_start_y - 80) / line_height_offset)
        if len(display_lines) > max_lines_to_show:
            display_lines = display_lines[-max_lines_to_show:]

        cursor_x, cursor_y_baseline = text_origin_x, text_origin_y

        if display_lines:
            last_displayed_line = display_lines[-1]

            trimmed_line = last_displayed_line.rstrip(' ')
            num_trailing_spaces = len(last_displayed_line) - len(trimmed_line)

            width_of_trimmed_part = cv2.getTextSize(trimmed_line, font, font_scale, font_thickness)[0][0]

            total_line_width = width_of_trimmed_part + (num_trailing_spaces * single_space_width)

            cursor_x = text_origin_x + total_line_width
            cursor_y_baseline = text_origin_y + (len(display_lines) - 1) * line_height_offset
        else:
            cursor_x = text_origin_x
            cursor_y_baseline = text_origin_y

        write_multiline_text(frame, '\n'.join(display_lines),
                             (text_origin_x, text_origin_y),
                             font, font_scale, text_color, font_thickness, cv2.LINE_AA, line_height_offset=line_height_offset)

        current_time = time.time()
        if current_time - last_cursor_toggle_time >= CURSOR_BLINK_RATE:
            cursor_on = not cursor_on
            last_cursor_toggle_time = current_time

        if cursor_on:
            cursor_height = int(cv2.getTextSize("A", font, font_scale, font_thickness)[0][1] * 1.2)
            cursor_thickness = 2

            cursor_y_top = cursor_y_baseline - cursor_height + int(cursor_height * 0.2)
            cursor_y_bottom = cursor_y_baseline + int(cursor_height * 0.2)

            cv2.line(frame, (cursor_x, cursor_y_top), (cursor_x, cursor_y_bottom), text_color, cursor_thickness)


        cv2.imshow('Virtual Camera Keyboard', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Virtual Keyboard application finished.")

if __name__ == "__main__":
    run_virtual_keyboard()