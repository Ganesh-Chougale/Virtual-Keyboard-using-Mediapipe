## Download Python if needed
Install Python version 3.9 or 3.10  

## Create enviroment:  
```bash
py -3.9 -m venv camera_keyboard_env
```  

## Activate Enviroment:  
Bash: 
```  
source camera_keyboard_env/Scripts/activate
```  
PowerShell:  
```bash
.\camera_keyboard_env\Scripts\Activate.ps1
```  

## install all Dependancies & Libraries:  
```bash
pip install opencv-python
pip install mediapipe
pip install pynput
```  

## Create Base python file & test:
```python
import cv2

def test_camera():
    # Try to open the default camera (usually index 0)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera opened successfully. Press 'q' to quit.")

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the frame
        cv2.imshow('Live Camera Feed', frame)

        # Wait for 1 millisecond and check if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print("Camera test finished.")

if __name__ == "__main__":
    test_camera()
```  
Open another terminal  
```bash
cd camera_keyboard_env/
python camera_test.py
```  

## Create Hand detecting file & test:  
```python
import cv2
import mediapipe as mp

def run_hand_detection():
    # --- Step 1: Integrate MediaPipe Hand Tracking ---
    # Initialize MediaPipe Hands solution
    # static_image_mode=False: Treat input images as a video stream (for better tracking).
    # max_num_hands=2: Detect up to two hands.
    # min_detection_confidence=0.5: Minimum confidence for hand detection.
    # min_tracking_confidence=0.5: Minimum confidence for hand tracking.
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Initialize MediaPipe Drawing utilities for visualizing landmarks
    mp_drawing = mp.solutions.drawing_utils

    # Open the default camera (usually index 0)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera and MediaPipe Hand Detection active. Press 'q' to quit.")

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Flip the frame horizontally for a more intuitive selfie-view
        # You can comment this out if you prefer the mirror image
        frame = cv2.flip(frame, 1)

        # --- Step 2: Process Frames for Hand Detection ---
        # Convert the BGR image frame (from OpenCV) to RGB (MediaPipe prefers RGB)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the RGB image with the MediaPipe Hands model
        # 'results' will contain detected hands and their landmarks if found
        results = hands.process(image_rgb)

        # --- Step 3 & 4: Extract Finger Landmarks & Visualize Detection ---
        # Initialize a list to store coordinates of chosen finger tips
        finger_tip_coords = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # --- Step 4: Visualize Detection ---
                # Draw hand skeleton and landmarks on the original BGR frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2), # Red dots
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2) # Green lines
                )

                # --- Step 3: Extract Finger Landmarks ---
                # Access specific landmarks. MediaPipe defines 21 landmarks per hand.
                # Here we'll focus on the index finger tip (landmark 8) and thumb tip (landmark 4).
                # The coordinates are normalized (0 to 1) relative to image width/height.
                # Convert normalized coordinates to pixel coordinates.

                # Index Finger Tip (Landmark 8)
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, c = frame.shape # Get image dimensions (height, width, channels)
                
                # Convert normalized coordinates to pixel coordinates
                # x = landmark.x * image_width
                # y = landmark.y * image_height
                
                index_tip_x = int(index_finger_tip.x * w)
                index_tip_y = int(index_finger_tip.y * h)
                finger_tip_coords.append((index_tip_x, index_tip_y))

                # --- Step 5: Basic Finger Tip Position Tracking ---
                # Draw a circle at the index finger tip and display its coordinates
                cv2.circle(frame, (index_tip_x, index_tip_y), 10, (255, 0, 0), cv2.FILLED) # Blue circle
                cv2.putText(frame, f'Index: ({index_tip_x}, {index_tip_y})',
                            (index_tip_x + 15, index_tip_y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # You can add similar logic for other finger tips if needed, e.g., thumb
                # thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                # thumb_tip_x = int(thumb_tip.x * w)
                # thumb_tip_y = int(thumb_tip.y * h)
                # cv2.circle(frame, (thumb_tip_x, thumb_tip_y), 10, (0, 255, 255), cv2.FILLED) # Yellow circle

        # Display the frame with annotations
        cv2.imshow('Hand Detection Feed', frame)

        # --- Exit Condition ---
        # Wait for 1 millisecond and check if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    hands.close() # Close the MediaPipe Hands model
    print("Hand detection test finished.")

if __name__ == "__main__":
    run_hand_detection()
```  
```bash
python hand_detector_test.py
```  

## Create Virtual Keyboard file & test:  
```python
import cv2
import mediapipe as mp
import time
import pynput.keyboard

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1, # Focus on single hand interaction for simplicity, can be increased
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# --- Interaction Logic Parameters ---
HOVER_DURATION_THRESHOLD = 0.5  # seconds a finger must stay on a key to trigger a press (reduced for responsiveness)
CURSOR_BLINK_RATE = 0.5       # seconds to toggle cursor visibility

# --- Keyboard Controller ---
keyboard_controller = pynput.keyboard.Controller()

# --- Global State Variables ---
current_hover_key = None
hover_start_time = None
last_pressed_key_char = None # Store character of the last pressed key to prevent rapid re-press
typed_text = ""
is_shift_active = False

# Cursor state variables
cursor_on = True
last_cursor_toggle_time = time.time()


# --- Virtual Keyboard Layout Definition ---
# These ratios define spacing and sizes relative to the screen dimensions
KEY_HEIGHT_RATIO = 0.12 # Height of each key as a fraction of frame height
KEY_WIDTH_RATIO = 0.075 # Width of each standard key as a fraction of frame width
HORIZONTAL_PADDING_RATIO = 0.005 # Padding between keys horizontally
VERTICAL_PADDING_RATIO = 0.015 # Padding between rows vertically

TEXT_AREA_HEIGHT_RATIO = 0.25 # Bottom 25% of the screen for text display

KEY_LAYOUT = [
    # Row 1 (Numbers and Symbols) - NEW ROW
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '='],
    # Row 2 (QWERTY)
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    # Row 3 (ASDFG)
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    # Row 4 (SHIFT, ZXCVB, Backspace)
    ['SHIFT', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', '<-'],
    # Row 5 (ENTER, SPACE)
    ['ENTER', 'SPACE']
]

# Mapping for shifted characters (e.g., '1' becomes '!') - NEW MAPPING
SHIFT_MAP = {
    '1': '!', '2': '@', '3': '#', '4': '$', '5': '%', '6': '^', '7': '&', '8': '*', '9': '(', '0': ')',
    '-': '_', '=': '+',
    # Add common punctuation if needed, e.g.,
    # '[': '{', ']': '}', ';': ':', "'": '"', ',': '<', '.': '>', '/': '?'
}

keys_on_screen = [] # This list will hold Key objects with their calculated pixel bounds

# --- Key Class Definition ---
class Key:
    def __init__(self, char, x, y, w, h):
        self.char = char
        self.x = x
        self.y = y
        self.width = w
        self.height = h
        self.is_hovered = False
        self.is_pressed = False # Flag for drawing feedback
        self.last_press_time = 0 # To prevent immediate re-press from holding finger

    def contains(self, px, py):
        # Check if the given point (px, py) is within the key's bounding box
        return self.x < px < self.x + self.width and \
               self.y < py < self.y + self.height

    def draw(self, frame, is_active=False, is_shift_key_active_global=False, SHIFT_MAP_REF=None):
        # Determine colors based on state
        fill_color = (200, 200, 200) # Default light gray
        if is_active:
            fill_color = (0, 255, 0) # Green if hovered or just pressed
        elif self.char == 'SHIFT' and is_shift_key_active_global:
            fill_color = (255, 200, 0) # Light blue for active SHIFT key

        border_color = (0, 0, 0) # Black border
        text_color = (0, 0, 0) # Black text

        # Draw key rectangle (filled and border)
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.width, self.y + self.height), fill_color, cv2.FILLED)
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.width, self.y + self.height), border_color, 2)

        # Determine text to display based on shift state - MODIFIED LOGIC
        display_char = self.char
        if display_char not in ['SPACE', '<-', 'SHIFT', 'ENTER']:
            if 'A' <= display_char <= 'Z': # It's an alphabet key
                display_char = display_char.upper() if is_shift_key_active_global else display_char.lower()
            elif SHIFT_MAP_REF and display_char in SHIFT_MAP_REF: # It's a numerical/symbol key with a shift counterpart
                display_char = SHIFT_MAP_REF[display_char] if is_shift_key_active_global else display_char
            # If it's a character like '/' or '.' that might not be in SHIFT_MAP and has no case, it stays as is.

        # Adjust text position for centering
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_size = cv2.getTextSize(display_char, font, font_scale, font_thickness)[0]
        text_x = self.x + (self.width - text_size[0]) // 2
        text_y = self.y + (self.height + text_size[1]) // 2
        cv2.putText(frame, display_char, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)


def calculate_key_positions(frame_width, frame_height):
    """
    Calculates the pixel coordinates and dimensions for each key based on frame dimensions.
    Keyboard is positioned at the top of the screen.
    """
    global keys_on_screen
    keys_on_screen = [] # Clear previous calculations

    keyboard_area_start_y = 0 # Keyboard starts from the top
    keyboard_area_height = int(frame_height * (1 - TEXT_AREA_HEIGHT_RATIO)) # Total height for keyboard area

    key_h = int(keyboard_area_height * KEY_HEIGHT_RATIO)
    key_w = int(frame_width * KEY_WIDTH_RATIO) # Standard key width

    horiz_padding = int(frame_width * HORIZONTAL_PADDING_RATIO)
    vert_padding = int(keyboard_area_height * VERTICAL_PADDING_RATIO) # Vertical padding within keyboard area

    y_offset = keyboard_area_start_y + vert_padding # Initial Y for the first row

    for row_index, row in enumerate(KEY_LAYOUT):
        current_row_keys_width = 0
        # Calculate row width to center it
        for char in row:
            char_key_w = key_w
            if char == 'SPACE':
                char_key_w = int(key_w * 5) # Example: Spacebar 5 times standard width
            elif char == 'SHIFT' or char == '<-' or char == 'ENTER':
                char_key_w = int(key_w * 1.8)
            current_row_keys_width += char_key_w + horiz_padding

        # Adjust for last key not having padding
        current_row_keys_width -= horiz_padding
        if current_row_keys_width < 0: # Handle empty row case
             current_row_keys_width = 0

        x_offset = int((frame_width - current_row_keys_width) / 2) # Center the row

        for col_index, char in enumerate(row):
            current_key_w = key_w
            if char == 'SPACE':
                current_key_w = int(key_w * 5)
            elif char == 'SHIFT' or char == '<-' or char == 'ENTER':
                current_key_w = int(key_w * 1.8)

            x = x_offset
            y = y_offset

            keys_on_screen.append(Key(char, x, y, current_key_w, key_h))
            x_offset += current_key_w + horiz_padding # Move x_offset for next key

        y_offset += key_h + vert_padding # Move to next row


def get_key_at_position(px, py):
    """
    Checks if a given point (px, py) is within any key's bounding box.
    Returns the Key object if found, otherwise None.
    """
    for key in keys_on_screen:
        if key.contains(px, py):
            return key
    return None

def write_multiline_text(img, text, org, font, fontScale, color, thickness, lineType, line_height_offset=30):
    """
    Writes multiline text on an image.
    This function no longer returns cursor position, it's calculated externally.
    """
    y0, dy = org[1], line_height_offset
    lines = text.split('\n') # text should already be wrapped into display_lines

    for i, line in enumerate(lines):
        y = y0 + i * dy
        cv2.putText(img, line, (org[0], y), font, fontScale, color, thickness, lineType)


def run_virtual_keyboard():
    global current_hover_key, hover_start_time, last_pressed_key_char, typed_text, is_shift_active
    global cursor_on, last_cursor_toggle_time

    cap = cv2.VideoCapture(0)

    # Set camera resolution to max available for full screen (may vary by webcam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Read the first frame to get actual frame dimensions after setting resolution
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read initial frame for dimensioning.")
        cap.release()
        return

    frame_height, frame_width, _ = frame.shape
    print(f"Camera opened with resolution: {frame_width}x{frame_height}")

    # Set the window to full screen. This must be done AFTER the camera is opened and frame size determined.
    cv2.namedWindow('Virtual Camera Keyboard', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Virtual Camera Keyboard', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Calculate key positions based on the actual frame dimensions
    calculate_key_positions(frame_width, frame_height)

    print("Virtual Keyboard active. Press 'q' to quit.")

    # Cache single space width for cursor calculation
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    single_space_width = cv2.getTextSize(" ", font, font_scale, font_thickness)[0][0]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera. Exiting.")
            break

        frame = cv2.flip(frame, 1) # Flip horizontally for selfie-view

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        finger_tip_coords = None # Store the index finger tip coordinates
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2), # Red dot
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2) # Green line
                )

                # Get index finger tip coordinates (Landmark 8)
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, c = frame.shape
                index_tip_x = int(index_finger_tip.x * w)
                index_tip_y = int(index_finger_tip.y * h)
                finger_tip_coords = (index_tip_x, index_tip_y) # Use the first detected finger

                # Draw a blue circle at the finger tip
                cv2.circle(frame, (index_tip_x, index_tip_y), 10, (255, 0, 0), cv2.FILLED)

        # --- Interaction Logic ---
        detected_key_this_frame = None
        if finger_tip_coords:
            detected_key_this_frame = get_key_at_position(finger_tip_coords[0], finger_tip_coords[1])

        # Reset hover state for all keys for drawing
        for key in keys_on_screen:
            key.is_hovered = False

        if detected_key_this_frame:
            detected_key_this_frame.is_hovered = True # Mark current key as hovered for drawing

            if current_hover_key is None:
                # Just started hovering over a new key
                current_hover_key = detected_key_this_frame
                hover_start_time = time.time()

            elif current_hover_key.char == detected_key_this_frame.char:
                # Still hovering over the same key
                if time.time() - hover_start_time >= HOVER_DURATION_THRESHOLD:
                    # Hover duration met, check if it's a new press (not the same key held down)
                    if last_pressed_key_char != current_hover_key.char:
                        current_hover_key.is_pressed = True # Mark for drawing feedback

                        # --- Simulate Keystroke ---
                        char_to_process = current_hover_key.char
                        final_char_to_send = None # Character that will be sent to pynput and added to typed_text

                        if char_to_process == 'SPACE':
                            keyboard_controller.press(pynput.keyboard.Key.space)
                            keyboard_controller.release(pynput.keyboard.Key.space)
                            final_char_to_send = ' '
                        elif char_to_process == '<-': # Backspace
                            keyboard_controller.press(pynput.keyboard.Key.backspace)
                            keyboard_controller.release(pynput.keyboard.Key.backspace)
                            typed_text = typed_text[:-1] if typed_text else "" # Remove last char
                        elif char_to_process == 'ENTER':
                            keyboard_controller.press(pynput.keyboard.Key.enter)
                            keyboard_controller.release(pynput.keyboard.Key.enter)
                            final_char_to_send = '\n' # Add newline to typed text display
                        elif char_to_process == 'SHIFT':
                            is_shift_active = not is_shift_active # Toggle shift state
                            final_char_to_send = None # SHIFT key doesn't type a character
                        else:
                            # Handle regular character (alphabet, number, symbol)
                            if is_shift_active:
                                # Determine the shifted character for display and pynput
                                if 'A' <= char_to_process <= 'Z': # Alphabet characters
                                    final_char_to_send = char_to_process.upper()
                                elif char_to_process in SHIFT_MAP: # Numerical/symbol characters with shift mappings
                                    final_char_to_send = SHIFT_MAP[char_to_process]
                                else: # Fallback for other characters that might not have explicit shift mapping or case
                                    final_char_to_send = char_to_process
                            else: # Shift is not active
                                # Alphabet characters should be lowercase
                                if 'A' <= char_to_process <= 'Z':
                                    final_char_to_send = char_to_process.lower()
                                else: # Numbers and unshifted symbols
                                    final_char_to_send = char_to_process
                            
                            # Send the final character to the system keyboard
                            if final_char_to_send: # Only press if there's a character to send
                                keyboard_controller.press(final_char_to_send)
                                keyboard_controller.release(final_char_to_send)


                        # Update typed_text for display purposes
                        if final_char_to_send is not None and char_to_process != '<-': # Backspace already handled
                             typed_text += final_char_to_send

                        last_pressed_key_char = current_hover_key.char
                        current_hover_key.last_press_time = time.time() # Record press time
                        # Reset cursor blink state on key press for immediate visibility
                        cursor_on = True
                        last_cursor_toggle_time = time.time()
            else:
                # Moved to a different key
                current_hover_key = detected_key_this_frame
                hover_start_time = time.time()
                last_pressed_key_char = None # Reset last pressed key when moving to a new one
        else:
            # No key is currently hovered over
            current_hover_key = None
            hover_start_time = None
            last_pressed_key_char = None # Allow pressing the same key again if finger lifts

        # Draw all keys on the frame
        for key in keys_on_screen:
            # Check if key was just pressed or is still hovered
            active_for_drawing = key.is_hovered or \
                                 (key.is_pressed and (time.time() - key.last_press_time < 0.2)) # Briefly highlight press
            key.draw(frame, active_for_drawing, is_shift_active, SHIFT_MAP) # Pass SHIFT_MAP to Key.draw
            key.is_pressed = False # Reset press state after drawing

        # --- Text Viewing Area at Bottom ---
        text_area_start_y = int(frame_height * (1 - TEXT_AREA_HEIGHT_RATIO))
        # Draw a translucent background for the text area
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, text_area_start_y), (frame_width, frame_height), (50, 50, 50), -1)
        alpha = 0.6 # Transparency factor
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Display typed text
        text_color = (255, 255, 255) # White text
        text_origin_x = 50
        text_origin_y = text_area_start_y + 40 # Adjust for some top padding
        line_height_offset = 30 # From the write_multiline_text function

        # --- Improved Text Wrapping Logic to preserve spaces for cursor ---
        display_lines = []
        current_line_buffer = ""
        avg_char_width = cv2.getTextSize("A", font, font_scale, font_thickness)[0][0]
        max_chars_per_line_est = int((frame_width - text_origin_x * 2) / avg_char_width)

        # Process the entire typed_text including explicit newlines
        segments = typed_text.split('\n')
        for seg_idx, segment in enumerate(segments):
            # Process each segment for word wrapping
            temp_segment_buffer = "" # Accumulate characters for a single line
            for char_idx, char in enumerate(segment):
                # Check if adding the next character exceeds the line limit
                # We need to get the actual width of the current_line_buffer + char
                # For simplicity, we use estimated char width for wrapping decisions.
                # The actual pixel width for cursor is calculated later.
                if len(temp_segment_buffer) + 1 > max_chars_per_line_est:
                    # Find the last space to wrap cleanly, or force break
                    last_space_idx = temp_segment_buffer.rfind(' ')
                    if last_space_idx > 0: # Found a space to break
                        display_lines.append(temp_segment_buffer[:last_space_idx])
                        temp_segment_buffer = temp_segment_buffer[last_space_idx+1:] # Start new line after space
                    else: # No space or space at beginning, force break
                        display_lines.append(temp_segment_buffer)
                        temp_segment_buffer = ""
                
                temp_segment_buffer += char

            # After processing all characters in a segment, add any remaining buffer
            if temp_segment_buffer:
                display_lines.append(temp_segment_buffer)
            
            # If it was an explicit newline, add an empty line (unless it's the very last segment)
            if seg_idx < len(segments) - 1:
                display_lines.append("")

        # Ensure there's at least one line to display the cursor on, even if text is empty
        if not display_lines:
            display_lines.append("")

        # If too many lines, only show the recent ones that fit
        max_lines_to_show = int((frame_height - text_area_start_y - 80) / line_height_offset) # 80 for top/bottom padding
        if len(display_lines) > max_lines_to_show:
            display_lines = display_lines[-max_lines_to_show:]
        
        # Calculate cursor position before drawing text - FIX FOR SPACES
        cursor_x, cursor_y_baseline = text_origin_x, text_origin_y
        
        if display_lines:
            last_displayed_line = display_lines[-1]
            
            # Custom calculation for line width including trailing spaces
            trimmed_line = last_displayed_line.rstrip(' ') # Get line without trailing spaces
            num_trailing_spaces = len(last_displayed_line) - len(trimmed_line)
            
            # Get width of the non-space part
            width_of_trimmed_part = cv2.getTextSize(trimmed_line, font, font_scale, font_thickness)[0][0]
            
            # Total width considering all spaces
            total_line_width = width_of_trimmed_part + (num_trailing_spaces * single_space_width)
            
            cursor_x = text_origin_x + total_line_width
            cursor_y_baseline = text_origin_y + (len(display_lines) - 1) * line_height_offset
        else: # If display_lines is somehow empty (shouldn't happen with the check above)
            cursor_x = text_origin_x
            cursor_y_baseline = text_origin_y
        
        # Draw the text
        write_multiline_text(frame, '\n'.join(display_lines),
                             (text_origin_x, text_origin_y),
                             font, font_scale, text_color, font_thickness, cv2.LINE_AA, line_height_offset=line_height_offset)

        # --- Cursor Blinking Logic and Drawing ---
        current_time = time.time()
        if current_time - last_cursor_toggle_time >= CURSOR_BLINK_RATE:
            cursor_on = not cursor_on
            last_cursor_toggle_time = current_time

        if cursor_on:
            # Draw cursor as a vertical line
            cursor_height = int(cv2.getTextSize("A", font, font_scale, font_thickness)[0][1] * 1.2) # 120% of 'A' height
            cursor_thickness = 2
            
            # The top of the cursor should be slightly above the baseline of the text
            # These values might need fine-tuning based on font and desired look
            cursor_y_top = cursor_y_baseline - cursor_height + int(cursor_height * 0.2)
            cursor_y_bottom = cursor_y_baseline + int(cursor_height * 0.2)

            cv2.line(frame, (cursor_x, cursor_y_top), (cursor_x, cursor_y_bottom), text_color, cursor_thickness)


        cv2.imshow('Virtual Camera Keyboard', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Virtual Keyboard application finished.")

if __name__ == "__main__":
    run_virtual_keyboard()
```  
```bash
python virtual_keyboard.py
```  