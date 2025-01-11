import cv2
import numpy as np
import pandas as pd
import random

# Constants for simulation
width, height = 1920, 1080
fps = 60
duration = 60  # seconds
total_frames = fps * duration
output_file = 'expanding_universe_with_galaxies_and_text.mp4'

# Parameters
degree_step = 20  # Degrees between lines
arc_step = 50  # Pixels between left-side arcs (based on horizontal displacement)
scale_multiplier = 12  # Adjusted multiplier for universe radius scaling
rgb_gray = (150, 150, 150)  # RGB values for gray color
galaxy_gray = (100, 100, 100)  # RGB values for gray galaxies in snapshots

# Universe data (generated previously)
data = pd.read_csv('Universe_Scale_Factor_Over_Time.csv')  # Replace with your DataFrame export
time_values = data['Time (Billion Years)'].values
scale_factors = data['Scale Factor (a(t))'].values

# Map scale factor to radius growth
max_radius = (min(width, height) // 4) * scale_multiplier  # Ensures max_radius <= 50
radii = (scale_factors / scale_factors[-1]) * max_radius  # Normalize and scale

# Interpolate radii to fit total_frames
frame_radii = np.interp(
    np.linspace(0, 1, total_frames),
    np.linspace(0, 1, len(radii)),
    radii
)
frame_times = np.interp(
    np.linspace(0, 1, total_frames),
    np.linspace(0, 1, len(time_values)),
    time_values
)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Initialize trail points for angles
angles = np.deg2rad(np.arange(1, 360, degree_step))  # Angles for left half-circle
trail_points = {angle: [] for angle in angles}

# List to store the X positions, radii, and galaxy snapshots where arcs are drawn
arc_snapshots = []  # Each entry: {'arc_x': int, 'arc_radius': int, 'galaxies': list of galaxy states}

start_x = 50
start_y = height // 2
end_x = width - 50

# ------------------- Galaxy Initialization -------------------

# Define the number of galaxies
num_galaxies = 1000

# Store galaxies with normalized positions and sizes
galaxies = []

# Maximum universe radius
initial_radius = radii[0] if radii[0] > 0 else 1  # Prevent division by zero

for _ in range(num_galaxies):
    # Random angle between 0 and 2π
    angle = random.uniform(0, 2 * np.pi)
    
    # Random normalized radius between 0 and 1
    radius_norm = random.uniform(0, 1)
    
    # Convert polar to Cartesian coordinates
    x_norm = np.cos(angle) * radius_norm
    y_norm = np.sin(angle) * radius_norm
    
    # Random initial size between 1 and 3 pixels
    size_initial = random.uniform(1, 3)  # Smaller sizes for better visibility
    size_norm = size_initial / max_radius  # Normalize size relative to max_radius
    
    # Assign a random color to each galaxy (optional)
    # For diversity, random colors can be used. Here, keeping it white.
    color = (255, 255, 255)
    
    galaxies.append({
        'x_norm': x_norm,
        'y_norm': y_norm,
        'size_norm': size_norm,
        'color': color,
        'trail': []  # Initialize trail for each galaxy
    })

# -------------------------------------------------------------

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
    if len(arc_snapshots) == 0 or current_x - arc_snapshots[-1]['arc_x'] >= arc_step:
        # Store the current arc position and radius
        arc_snapshot = {
            'arc_x': current_x,
            'arc_radius': radius,
            'galaxies': []  # To store the current state of galaxies
        }
        
        # Capture the current state of all galaxies
        for galaxy in galaxies:
            # Scale galaxy positions
            scaled_x = int(galaxy['x_norm'] * frame_radii[frame_idx])
            scaled_y = int(galaxy['y_norm'] * frame_radii[frame_idx])
            
            # Scale galaxy size
            scaled_size = max(int(galaxy['size_norm'] * frame_radii[frame_idx]), 1)  # Ensure at least size 1
            
            # Calculate actual position on the frame
            galaxy_position = (current_x + scaled_x, start_y - scaled_y)
            
            # Store the scaled position and size
            arc_snapshot['galaxies'].append({
                'position': galaxy_position,
                'size': scaled_size,
                'color': galaxy['color']
            })
        
        arc_snapshots.append(arc_snapshot)  # Add to snapshots

    # Update trail points for all angles
    for angle in trail_points.keys():
        x_offset = int(radius * np.cos(angle))
        y_offset = int(radius * np.sin(angle))
        point = (current_x + x_offset, start_y - y_offset)
        trail_points[angle].append(point)

    # Draw the expanding circle as an outline (white for the current state)
    cv2.circle(frame, (current_x, start_y), radius, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw the trails for the left half-circle (gray for past states)
    for angle, points in trail_points.items():
        for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], rgb_gray, 1, cv2.LINE_AA)

    # Draw arcs and their galaxy snapshots at positions stored in `arc_snapshots` (gray for past states)
    for snapshot in arc_snapshots:
        arc_x = snapshot['arc_x']
        arc_radius = snapshot['arc_radius']
        
        # Draw the universe snapshot circle
        cv2.circle(
            frame,
            (arc_x, start_y),
            arc_radius,
            rgb_gray,
            1,
            cv2.LINE_AA,
        )
        
        # Draw the corresponding galaxy snapshot
        for galaxy_snapshot in snapshot['galaxies']:
            galaxy_pos = galaxy_snapshot['position']
            galaxy_size = galaxy_snapshot['size']
            galaxy_color = galaxy_gray  # Use gray color for snapshots
            
            # Draw the galaxy as a filled circle
            cv2.circle(frame, galaxy_pos, galaxy_size, galaxy_color, -1, cv2.LINE_AA)

    # ------------------- Draw Galaxies -------------------

    # Calculate the scaling factor relative to the initial max_radius
    scale_factor = frame_radii[frame_idx] / max_radius if max_radius != 0 else 1

    for galaxy in galaxies:
        # Scale galaxy positions
        scaled_x = int(galaxy['x_norm'] * frame_radii[frame_idx])
        scaled_y = int(galaxy['y_norm'] * frame_radii[frame_idx])
        
        # Scale galaxy size
        scaled_size = max(int(galaxy['size_norm'] * frame_radii[frame_idx]), 1)  # Ensure at least size 1
        
        # Calculate actual position on the frame
        galaxy_position = (current_x + scaled_x, start_y - scaled_y)
        
        # Update galaxy trail
        galaxy['trail'].append(galaxy_position)
        if len(galaxy['trail']) > 30:  # Limit trail length
            galaxy['trail'].pop(0)
        
        # Draw the galaxy trail
        for i in range(1, len(galaxy['trail'])):
            cv2.line(frame, galaxy['trail'][i - 1], galaxy['trail'][i], rgb_gray, 1, cv2.LINE_AA)
        
        # Draw the galaxy as a filled circle
        cv2.circle(frame, galaxy_position, scaled_size, galaxy['color'], -1, cv2.LINE_AA)
    
    # ------------------------------------------------------

    # Add text below the circle (time)
    text_time = f"{current_time:.2f} Billones de años"
    text_size_time = cv2.getTextSize(text_time, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x_time = current_x - text_size_time[0] // 2
    text_y_time = start_y + radius + 40
    cv2.putText(
        frame,
        text_time,
        (text_x_time, text_y_time),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Add text to the top of the circle (universe expansion size with units)
    text_size = f"{radius:.0f} a(t)"
    text_size_universe = cv2.getTextSize(text_size, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x_universe = current_x - text_size_universe[0] // 2
    text_y_universe = start_y - radius - 20
    cv2.putText(
        frame,
        text_size,
        (text_x_universe, text_y_universe),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Draw a horizontal timeline
    cv2.line(frame, (0, start_y), (width, start_y), (255, 255, 255), 2, cv2.LINE_AA)

    # ------------------- Draw Universe Trails -------------------
    
    # Note: The universe trails and their galaxy snapshots are already handled above with `trail_points` and `arc_snapshots`.
    
    # -------------------------------------------------------------

    # Display the frame (optional, can be commented out if not needed)
    cv2.imshow('Expanding Universe with Galaxies and Text', frame)

    # Write the frame to the video
    out.write(frame)

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video writer and close display window
out.release()
cv2.destroyAllWindows()
print(f"Simulation saved to {output_file}")
