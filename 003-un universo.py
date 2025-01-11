import cv2
import numpy as np

# Constants for simulation
width, height = 1920, 1080
fps = 60
duration = 60  # seconds
total_frames = fps * duration
output_file = 'expanding_universe.mp4'

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Simulation parameters
start_x = 50
start_y = height // 2
end_x = width - 50
max_radius = min(width, height) // 4

for frame_idx in range(total_frames):
    # Calculate progression (0 to 1)
    t = frame_idx / total_frames

    # Create a black canvas
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate the current position and radius
    current_x = int(start_x + t * (end_x - start_x))
    radius = int(t * max_radius)

    # Draw the expanding circle
    cv2.circle(frame, (current_x, start_y), radius, (255, 0, 0), -1)

    # Draw a horizontal timeline
    cv2.line(frame, (0, start_y), (width, start_y), (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Expanding Universe Simulation', frame)

    # Write the frame to the video
    out.write(frame)

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video writer and close display window
out.release()
cv2.destroyAllWindows()
print(f"Simulation saved to {output_file}")
