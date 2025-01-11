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
center_x = width // 2
center_y = height // 2
max_radius = min(center_x, center_y) - 50
max_displacement = width - 100

for frame_idx in range(total_frames):
    # Calculate progression (0 to 1)
    t = frame_idx / total_frames

    # Create a black canvas
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate the current radius and horizontal displacement
    radius = int(t * max_radius)
    displacement = int(t * max_displacement)

    # Draw the expanding circle (Big Bang explosion simulation)
    cv2.circle(frame, (center_x - displacement, center_y), radius, (255, 0, 0), -1)

    # Draw the current state circle (present universe)
    cv2.circle(frame, (center_x + displacement, center_y), radius, (0, 255, 0), -1)

    # Draw a horizontal timeline
    cv2.line(frame, (0, center_y), (width, center_y), (255, 255, 255), 2)

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
