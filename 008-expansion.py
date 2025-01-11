import cv2
import numpy as np
import pandas as pd

# Constants for simulation
width, height = 1920, 1080
fps = 60
duration = 60  # seconds
total_frames = fps * duration
output_file = 'expanding_universe_with_params.mp4'

# Parameters
degree_step = 20  # Degrees between lines
arc_step = 50  # Pixels between left-side arcs
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

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Initialize trail points for angles
angles = np.deg2rad(np.arange(-90, 91, degree_step))  # Angles for left half-circle
trail_points = {angle: [] for angle in angles}

start_x = 50
start_y = height // 2
end_x = width - 50

for frame_idx in range(total_frames):
    # Calculate progression (0 to 1)
    t = frame_idx / total_frames

    # Create a black canvas
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate the current position and radius
    current_x = int(start_x + t * (end_x - start_x))
    radius = int(frame_radii[frame_idx])

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

    # Draw arcs (left side of the circle) every `arc_step` pixels
    for offset in range(0, radius, arc_step):
        cv2.ellipse(
            frame,
            (current_x, start_y),
            (offset, offset),
            0, 90, 270,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Draw a horizontal timeline
    cv2.line(frame, (0, start_y), (width, start_y), (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Expanding Universe with Parameters', frame)

    # Write the frame to the video
    out.write(frame)

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video writer and close display window
out.release()
cv2.destroyAllWindows()
print(f"Simulation saved to {output_file}")
