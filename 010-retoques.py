import cv2
import numpy as np
import pandas as pd

# Constants for simulation
width, height = 1920, 1080
fps = 60
duration = 60  # seconds
total_frames = fps * duration
output_file = 'expanding_universe_with_text.mp4'

# Parameters
degree_step = 20  # Degrees between lines
arc_step = 50  # Pixels between left-side arcs (based on horizontal displacement)
scale_multiplier = 4.0  # Multiplier for universe radius scaling

# Universe data (generated previously)
data = pd.read_csv('Universe_Scale_Factor_Over_Time.csv')  # Replace with your DataFrame export
time_values = data['Time (Billion Years)'].values
scale_factors = data['Scale Factor (a(t))'].values

# Map scale factor to radius growth
max_radius = min(width, height) // 4 * scale_multiplier
radii = (scale_factors / scale_factors[-1]) * max_radius  # Normalize and scale

# Interpolate radii to fit total_frames
frame_radii = np.interp(np.linspace(0, 1, total_frames), np.linspace(0, 1, len(radii)), radii)
frame_times = np.interp(np.linspace(0, 1, total_frames), np.linspace(0, 1, len(time_values)), time_values)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Initialize trail points for angles
angles = np.deg2rad(np.arange(-90, 91, degree_step))  # Angles for left half-circle
trail_points = {angle: [] for angle in angles}

# List to store the X positions and radii where arcs are drawn
arc_positions = []  # Each entry: (x_position, radius)

start_x = 50
start_y = height // 2
end_x = width - 50

for frame_idx in range(total_frames):
    # Calculate progression (0 to 1)
    t = frame_idx / total_frames

    # Create a black canvas
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate the current position, radius, and time
    current_x = int(start_x + t * (end_x - start_x))
    radius = int(frame_radii[frame_idx])
    current_time = frame_times[frame_idx]

    # Check if a new arc needs to be drawn based on displacement
    if len(arc_positions) == 0 or current_x - arc_positions[-1][0] >= arc_step:
        arc_positions.append((current_x, radius))  # Store position and radius

    # Update trail points for all angles (-90 to 90)
    for angle in trail_points.keys():
        x_offset = int(radius * np.cos(angle))
        y_offset = int(radius * np.sin(angle))
        point = (current_x + x_offset, start_y - y_offset)
        trail_points[angle].append(point)

    # Draw the expanding circle as an outline
    cv2.circle(frame, (current_x, start_y), radius, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw the trails for the left half-circle
    for angle, points in trail_points.items():
        for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], (255, 255, 255), 1, cv2.LINE_AA)

    # Draw arcs at positions stored in `arc_positions`
    for arc_x, arc_radius in arc_positions:
        cv2.ellipse(
            frame,
            (arc_x, start_y),
            (arc_radius, arc_radius),  # Use stored radius
            0, 90, 270,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Add text to the center of the circle (time)
    text_time = f"Time: {current_time:.2f} Billion Years"
    text_size_time = cv2.getTextSize(text_time, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x_time = current_x - text_size_time[0] // 2
    text_y_time = start_y + text_size_time[1] // 2
    cv2.putText(frame, text_time, (text_x_time, text_y_time), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Add text to the top of the circle (universe expansion size)
    text_size = f"Size: {radius:.0f}"
    text_size_universe = cv2.getTextSize(text_size, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x_universe = current_x - text_size_universe[0] // 2
    text_y_universe = start_y - radius - 20
    cv2.putText(frame, text_size, (text_x_universe, text_y_universe), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw a horizontal timeline
    cv2.line(frame, (0, start_y), (width, start_y), (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Expanding Universe with Text', frame)

    # Write the frame to the video
    out.write(frame)

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video writer and close display window
out.release()
cv2.destroyAllWindows()
print(f"Simulation saved to {output_file}")
