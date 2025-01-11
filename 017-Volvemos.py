import cv2
import numpy as np
import pandas as pd
import random
import time

# ------------------------- Configuration -------------------------

# Constants for simulation
width, height = 1920, 1080
fps = 60
duration = 30  # seconds
total_frames = fps * duration
epoch = round(time.time())
output_file = 'jocarsa_tragaluz_cuantico_expansion_galaxias_'+str(epoch)+'.mp4'

# Parameters
degree_step = 20  # Degrees between lines
arc_step = 50  # Pixels between left-side arcs (based on horizontal displacement)
scale_multiplier = 12  # Adjusted multiplier for universe radius scaling
rgb_gray = (150, 150, 150)  # RGB values for gray color
galaxy_gray = (100, 100, 100)  # RGB values for gray galaxies in snapshots

# Bloom Effect Parameters
bloom_intensity = 2  # Amount of bloom effect (increase for stronger glow)
bloom_blur_radius = 15  # Must be a positive odd integer

# Motion Blur Parameters
motion_blur_opacity = 0.2  # Opacity for motion blur overlay (higher for more blur)

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
num_galaxies = 500

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

# ------------------------- Bloom Effect Function -------------------------

def apply_bloom(frame, intensity=0.3, blur_radius=15):
    """
    Apply a bloom (glow) effect to the given frame.

    Parameters:
        frame (numpy.ndarray): The original frame in BGR format.
        intensity (float): The amount of bloom effect. Typical range: 0.0 (no bloom) to 2.0+.
        blur_radius (int): The radius for the box blur (must be a positive odd integer).

    Returns:
        numpy.ndarray: The frame with the bloom effect applied.
    """
    # Ensure blur_radius is a positive odd integer
    if blur_radius <= 0 or blur_radius % 2 == 0:
        print(f"Invalid BLUR_RADIUS: {blur_radius}. It must be a positive odd integer. Resetting to 15.")
        blur_radius = 15

    # Duplicate the main frame
    blurred_frame = frame.copy()

    # Apply a box blur to the duplicated frame
    blurred_frame = cv2.blur(blurred_frame, (blur_radius, blur_radius))

    # Scale the blurred frame by intensity
    blurred_frame = cv2.convertScaleAbs(blurred_frame, alpha=intensity, beta=0)

    # Composite the blurred frame over the original frame using ADD operation
    bloom_frame = cv2.add(frame, blurred_frame)

    return bloom_frame

# ------------------------- Motion Blur Function -------------------------

def apply_motion_blur(accumulated_frame, current_frame, opacity=0.2):
    """
    Apply a motion blur effect by blending the accumulated frame with the current frame.

    Parameters:
        accumulated_frame (numpy.ndarray): The frame accumulating previous frames.
        current_frame (numpy.ndarray): The newly drawn current frame.
        opacity (float): The opacity of the accumulated frame in the blend.

    Returns:
        numpy.ndarray: The blended frame with motion blur applied.
    """
    # Blend the accumulated frame with the current frame
    blended_frame = cv2.addWeighted(accumulated_frame, opacity, current_frame, 1 - opacity, 0)
    return blended_frame

# ------------------------- Helper Functions -------------------------

def get_scaled_position(position, center_x=start_x, center_y=start_y, scale=1.0):
    """Scale the position relative to the center based on the current scale."""
    scaled_x = int(center_x + (position[0]) * scale)
    scaled_y = int(center_y - (position[1]) * scale)  # Inverted Y-axis for correct orientation
    return (scaled_x, scaled_y)

def get_scaled_radius(radius, scale=1.0):
    """Scale the radius based on the current scale, ensuring it's at least 1."""
    return max(int(radius * scale), 1)

def compute_displacement(particle_position, scale=1.0):
    """
    Placeholder function for displacement based on particle position.
    Can be customized to add more dynamic effects.
    """
    # For simplicity, returning zero displacement
    return np.array([0.0, 0.0])

# ------------------------- Accumulation Frame Initialization -------------------------

# Initialize an accumulation frame for motion blur as a black image
accumulated_frame = np.zeros((height, width, 3), dtype=np.uint8)

# ------------------------- Simulation Loop -------------------------

for frame_idx in range(total_frames):
    # Calculate progression (0 to 1)
    t = frame_idx / total_frames

    # ------------------------- Create a Black Canvas -------------------------
    # Note: We start with a black frame each time
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate the current position, radius, and time
    current_x = int(start_x + t * (end_x - start_x))
    radius = int(frame_radii[frame_idx])
    current_time = frame_times[frame_idx]
    # Draw a horizontal timeline
    cv2.line(frame, (0, start_y), (width, start_y), (100, 100, 100), 1, cv2.LINE_AA)
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

    # ------------------- Apply Bloom Effect -------------------
    frame = apply_bloom(frame, intensity=bloom_intensity, blur_radius=bloom_blur_radius)
    # ------------------------------------------------------------

    # ------------------- Update Accumulation Frame for Motion Blur -------------------
    # Blend the accumulated frame with the current frame to create motion blur
    accumulated_frame = apply_motion_blur(accumulated_frame, frame, opacity=motion_blur_opacity)
    # Replace the current frame with the blended frame
    frame = accumulated_frame.copy()
    # ------------------------------------------------------------

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

    

    # ------------------- Draw Universe Trails -------------------
    
    # Note: The universe trails and their galaxy snapshots are already handled above with `trail_points` and `arc_snapshots`.
    
    # -------------------------------------------------------------

    # Write the frame to the video
    out.write(frame)

    # Display the frame (optional, can be commented out if not needed)
    cv2.imshow('Expanding Universe with Galaxies, Bloom, and Motion Blur', frame)

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------------- Cleanup -------------------------

# Release the video writer and close display window
out.release()
cv2.destroyAllWindows()
print(f"Simulation saved to {output_file}")
