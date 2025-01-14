import cv2
import numpy as np
import pandas as pd
import random
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------- Configuration -------------------------

# Constants for simulation
width, height = 1920, 1080
fps = 60
duration = 30  # seconds
total_frames = fps * duration
epoch = round(time.time())
output_file = 'jocarsa_tragaluz_cuantico_expansion_galaxias_' + str(epoch) + '.mp4'

# Parameters
degree_step = 20  # Degrees between lines
arc_step = 50  # Pixels between left-side arcs (based on horizontal displacement)
scale_multiplier_default = 12  # Adjusted multiplier for universe radius scaling
rgb_gray = (100, 100, 100)  # RGB values for general gray trails
galaxy_gray = (100, 100, 100)  # RGB values for gray galaxies in snapshots
particle_trail_color = (255, 255, 255)  # White color for particle trails (not used anymore)

# Bloom Effect Parameters
bloom_intensity = 2  # Amount of bloom effect (increase for stronger glow)
bloom_blur_radius = 15  # Must be a positive odd integer

# Motion Blur Parameters
motion_blur_opacity = 0.02  # Opacity for motion blur overlay (higher for more blur)

# Multithreading Parameters
bucket_width, bucket_height = 64, 64  # Size of each bucket
num_threads = 12  # Number of threads to use

# Universe data (generated previously)
# Ensure that 'Universe_Scale_Factor_Over_Time.csv' exists in the working directory
data = pd.read_csv('Universe_Scale_Factor_Over_Time.csv')  # Replace with your DataFrame export
time_values = data['Time (Billion Years)'].values
scale_factors = data['Scale Factor (a(t))'].values

# Map scale factor to radius growth
max_radius = (min(width, height) // 4) * scale_multiplier_default  # Ensures max_radius <= 50
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
arc_snapshots = []  # Each entry: {'arc_x': int, 'arc_radius': int, 'galaxies': list of galaxy states, 'particle_trails': list of particle states}

start_x = 50
start_y = height // 2
end_x = width - 50

# ------------------- Galaxy Initialization -------------------

# Define the number of galaxies
num_galaxies = 50

# Store galaxies with normalized positions and sizes
galaxies = []

# Parameters for Particle System
num_arms = 2  # Number of spiral arms per galaxy
particles_per_arm = 1500  # Number of particles per arm
particle_size_min = 1  # Minimum size of particles
particle_size_max = 3  # Maximum size of particles
particle_system_fraction = 0.1  # Fraction of universe radius for particle system

# Rotation and Scale Parameters
min_rotation_rate = -0.005  # Minimum rotation speed (radians per frame)
max_rotation_rate = -0.02   # Maximum rotation speed (radians per frame)
min_scale_multiplier = 0.8  # Minimum scale multiplier
max_scale_multiplier = 1.2  # Maximum scale multiplier

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
    
    # Initialize particle system for the galaxy
    particles = []
    arm_offset = (2 * np.pi) / num_arms  # Offset between arms

    for arm in range(num_arms):
        for p in range(particles_per_arm):
            # Distribute particles along the arm with increasing angle and distance
            theta = (p / particles_per_arm) * (4 * np.pi)  # Adjust for tighter or looser spirals
            r = (p / particles_per_arm)  # Normalize distance from center

            # Add arm offset
            theta += arm * arm_offset

            # Introduce random position variation for realism
            position_variation_angle = random.uniform(-0.2, 0.2)  # Small angular variation
            position_variation_radius = random.uniform(-0.1, 0.1)  # Small radial variation

            # Apply variation
            theta += position_variation_angle
            r += position_variation_radius

            # Ensure r stays within [0,1]
            r = max(0, min(r, 1))

            # Convert polar to Cartesian coordinates for particle position
            x_p = r * math.cos(theta)
            y_p = r * math.sin(theta)
            
            # Assign a random size to each particle
            particle_size_norm = random.uniform(particle_size_min, particle_size_max) / max_radius
            
            # Assign a unique random color to each particle
            particle_color = (
                random.randint(50, 255),  # Red component
                random.randint(50, 255),  # Green component
                random.randint(50, 255)   # Blue component
            )

            particles.append({
                'angle': theta,
                'distance_norm': r,
                'position_norm': (x_p, y_p),
                'size_norm': particle_size_norm,  # Add normalized size
                'color': particle_color  # Add unique color per particle
            })
    
    # Random Initial Rotation
    initial_rotation = random.uniform(0, 2 * np.pi)
    
    # Random Rotation Rate
    rotation_rate = random.uniform(min_rotation_rate, max_rotation_rate)
    
    # Current Rotation starts as initial_rotation
    current_rotation = initial_rotation
    
    # Random Scale Multiplier
    scale_multiplier = random.uniform(min_scale_multiplier, max_scale_multiplier)
    
    galaxies.append({
        'x_norm': x_norm,
        'y_norm': y_norm,
        'size_norm': size_norm,
        'color': color,
        'trail': [],  # Initialize trail for each galaxy
        'particles': particles,  # Add particle system
        'initial_rotation': initial_rotation,
        'rotation_rate': rotation_rate,
        'current_rotation': current_rotation,
        'scale_multiplier': scale_multiplier
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

# ------------------------- Bucket Generation Function -------------------------

def generate_buckets(width, height, bucket_width, bucket_height):
    """
    Generate a list of bucket coordinates covering the entire frame.

    Parameters:
        width (int): Width of the frame.
        height (int): Height of the frame.
        bucket_width (int): Width of each bucket.
        bucket_height (int): Height of each bucket.

    Returns:
        List of tuples: Each tuple contains (x_start, y_start, x_end, y_end) for a bucket.
    """
    buckets = []
    for y in range(0, height, bucket_height):
        for x in range(0, width, bucket_width):
            x_end = min(x + bucket_width, width)
            y_end = min(y + bucket_height, height)
            buckets.append((x, y, x_end, y_end))
    return buckets

# ------------------------- Worker Function -------------------------

def render_bucket(bucket_coords, shared_data):
    """
    Render a specific bucket region of the frame.

    Parameters:
        bucket_coords (tuple): (x_start, y_start, x_end, y_end) defining the bucket region.
        shared_data (dict): Shared data required for rendering.

    Returns:
        tuple: (x_start, y_start, rendered_bucket)
    """
    x_start, y_start, x_end, y_end = bucket_coords
    bucket_width = x_end - x_start
    bucket_height = y_end - y_start

    # Create a blank image for the bucket
    bucket_image = np.zeros((bucket_height, bucket_width, 3), dtype=np.uint8)

    # Unpack shared data
    galaxies = shared_data['galaxies']
    current_x = shared_data['current_x']
    start_y = shared_data['start_y']
    frame_radii = shared_data['frame_radii']
    frame_idx = shared_data['frame_idx']
    particle_system_fraction = shared_data['particle_system_fraction']
    bucket_image_rgb_gray = shared_data['rgb_gray']
    bucket_image_galaxy_gray = shared_data['galaxy_gray']
    particle_overlay = shared_data['particle_overlay']

    # Draw galaxy trails and particles within the bucket
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
            pt1 = galaxy['trail'][i - 1]
            pt2 = galaxy['trail'][i]
            # Check if the line segment is within the current bucket
            if (x_start <= pt1[0] < x_end and y_start <= pt1[1] < y_end) or \
               (x_start <= pt2[0] < x_end and y_start <= pt2[1] < y_end):
                cv2.line(bucket_image, (pt1[0] - x_start, pt1[1] - y_start),
                         (pt2[0] - x_start, pt2[1] - y_start),
                         bucket_image_rgb_gray, 1, cv2.LINE_AA)

        # Draw the galaxy as a filled circle if it falls within the bucket
        if x_start <= galaxy_position[0] < x_end and y_start <= galaxy_position[1] < y_end:
            cv2.circle(bucket_image,
                       (galaxy_position[0] - x_start, galaxy_position[1] - y_start),
                       scaled_size,
                       galaxy['color'],
                       -1,
                       cv2.LINE_AA)

        # Draw particle system
        for particle in galaxy['particles']:
            # Scale particle distance based on dynamic particle_system_radius and galaxy's scale multiplier
            particle_distance = particle['distance_norm'] * frame_radii[frame_idx] * galaxy['scale_multiplier']

            # Apply current rotation to particle position
            x_p, y_p = particle['position_norm']
            cos_theta = math.cos(galaxy['current_rotation'])
            sin_theta = math.sin(galaxy['current_rotation'])
            rotated_x = x_p * cos_theta - y_p * sin_theta
            rotated_y = x_p * sin_theta + y_p * cos_theta

            # Compute final particle position relative to the galaxy
            particle_x = galaxy_position[0] + int(rotated_x * particle_distance)
            particle_y = galaxy_position[1] - int(rotated_y * particle_distance)  # Inverted Y-axis

            # Scale particle size based on universe size
            scaled_particle_size = max(int(particle['size_norm'] * frame_radii[frame_idx]), 1)

            # Check if particle is within the current bucket
            if x_start <= particle_x < x_end and y_start <= particle_y < y_end:
                cv2.circle(bucket_image,
                           (particle_x - x_start, particle_y - y_start),
                           scaled_particle_size,
                           particle['color'],
                           -1,
                           cv2.LINE_AA)

    return (x_start, y_start, bucket_image)

# ------------------------- Simulation Loop with Multithreading -------------------------

# Generate all bucket coordinates once
all_buckets = generate_buckets(width, height, bucket_width, bucket_height)

# Initialize an accumulation frame for motion blur as a black image
accumulated_frame = np.zeros((height, width, 3), dtype=np.uint8)

# Initialize ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=num_threads)

for frame_idx in range(total_frames):
    # Calculate progression (0 to 1)
    t = frame_idx / total_frames

    # ------------------------- Create a Black Canvas -------------------------
    # Start with a black frame each time
    # Note: The frame will be constructed by combining all buckets
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Initialize an overlay for particle trails
    particle_overlay = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate the current position, radius, and time
    current_x = int(start_x + t * (end_x - start_x))
    radius = int(frame_radii[frame_idx])
    current_time = frame_times[frame_idx]
    
    # Calculate the dynamic particle system radius based on universe size
    # Define particle_system_radius as a fraction of the current universe radius
    particle_system_radius = frame_radii[frame_idx] * particle_system_fraction

    # Draw a horizontal timeline
    cv2.line(frame, (0, start_y), (width, start_y), (100, 100, 100), 1, cv2.LINE_AA)
    
    # Check if a new arc needs to be drawn based on displacement
    if len(arc_snapshots) == 0 or current_x - arc_snapshots[-1]['arc_x'] >= arc_step:
        # Store the current arc position and radius
        arc_snapshot = {
            'arc_x': current_x,
            'arc_radius': radius,
            'galaxies': [],  # To store the current state of galaxies
            'particle_trails': []  # To store particle trail states
        }
        
        # Capture the current state of all galaxies and their particles
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
            
            # Capture the particle system state for trails
            particle_trail = []
            for particle in galaxy['particles']:
                # Scale particle distance based on dynamic particle_system_radius and galaxy's scale multiplier
                particle_distance = particle['distance_norm'] * particle_system_radius * galaxy['scale_multiplier']
                
                # Apply current rotation to particle position
                x_p, y_p = particle['position_norm']
                cos_theta = math.cos(galaxy['current_rotation'])
                sin_theta = math.sin(galaxy['current_rotation'])
                rotated_x = x_p * cos_theta - y_p * sin_theta
                rotated_y = x_p * sin_theta + y_p * cos_theta
                
                # Compute final particle position relative to the galaxy
                particle_x = galaxy_position[0] + int(rotated_x * particle_distance)
                particle_y = galaxy_position[1] - int(rotated_y * particle_distance)  # Inverted Y-axis
                
                # Store particle position and color
                particle_trail.append({
                    'position': (particle_x, particle_y),
                    'color': particle['color']
                })
            
            arc_snapshot['particle_trails'].append(particle_trail)
        
        arc_snapshots.append(arc_snapshot)  # Add to snapshots

    # Update trail points for all angles
    for angle in trail_points.keys():
        x_offset = int(radius * np.cos(angle))
        y_offset = int(radius * np.sin(angle))
        point = (current_x + x_offset, start_y - y_offset)
        trail_points[angle].append(point)


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
        
        # Draw particle trails for this arc snapshot on the overlay
        for particle_trail in snapshot['particle_trails']:
            for particle in particle_trail:
                pos = particle['position']
                color = particle['color']
                if 0 <= pos[0] < width and 0 <= pos[1] < height:
                    particle_overlay[pos[1], pos[0]] = color  # Directly modify the overlay

    # Blend the particle_overlay with the main frame using alpha=0.1
    alpha = 0.1  # Transparency factor
    frame = cv2.addWeighted(frame, 1.0, particle_overlay, alpha, 0)

    # ------------------- Update Galaxy Rotations -------------------
    for galaxy in galaxies:
        # Update the current rotation based on rotation rate
        galaxy['current_rotation'] += galaxy['rotation_rate']
        # Keep the rotation angle within 0 to 2π
        galaxy['current_rotation'] = galaxy['current_rotation'] % (2 * np.pi)
    # ------------------------------------------------------------

    # ------------------- Prepare Shared Data for Threads -------------------
    shared_data = {
        'galaxies': galaxies,
        'current_x': current_x,
        'start_y': start_y,
        'frame_radii': frame_radii,
        'frame_idx': frame_idx,
        'particle_system_fraction': particle_system_fraction,
        'rgb_gray': rgb_gray,
        'galaxy_gray': galaxy_gray,
        'particle_overlay': particle_overlay
    }

    # ------------------- Multithreading Rendering -------------------
    # Submit all bucket rendering tasks to the executor
    futures = []
    for bucket in all_buckets:
        futures.append(executor.submit(render_bucket, bucket, shared_data))
    
    # Collect the rendered buckets and place them into the frame
    for future in as_completed(futures):
        x_start, y_start, bucket_image = future.result()
        x_end = x_start + bucket_image.shape[1]
        y_end = y_start + bucket_image.shape[0]
        frame[y_start:y_end, x_start:x_end] = cv2.add(frame[y_start:y_end, x_start:x_end], bucket_image)

    # ------------------- Draw Galaxies and Their Particle Systems -------------------
    # Note: This part is handled within the `render_bucket` function
    # If additional drawing is needed, it can be added here.

    # Draw the expanding circle as an outline (white for the current state)
    cv2.circle(frame, (current_x, start_y), radius, (255, 255, 255), 2, cv2.LINE_AA)

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
    text_size_str = f"{radius:.0f} a(t)"
    text_size_universe = cv2.getTextSize(text_size_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x_universe = current_x - text_size_universe[0] // 2
    text_y_universe = start_y - radius - 20
    cv2.putText(
        frame,
        text_size_str,
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
    cv2.imshow('Expanding Universe with Galaxies, Particles, Bloom, and Motion Blur', frame)

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------------- Cleanup -------------------------

# Shutdown the executor
executor.shutdown()

# Release the video writer and close display window
out.release()
cv2.destroyAllWindows()
print(f"Simulation saved to {output_file}")
