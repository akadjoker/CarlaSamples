import carla
import cv2
import numpy as np
import time
import random
import sys
import os
import datetime
import math
import argparse

# Global variables for dataset control
dataset_dir = None
dataset_images_dir = None
dataset_file = None
frame_count = 0
current_image = None

# Display settings
WIDTH = 800
HEIGHT = 600
FPS = 30

def create_dataset_session():
    """Creates a new session to store data (images and steering values)"""
    global dataset_dir, dataset_images_dir, frame_count, dataset_file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = f"dataset/session_{timestamp}"
    dataset_images_dir = f"{dataset_dir}/images"
    os.makedirs(dataset_images_dir, exist_ok=True)
    
    dataset_file = open(f"{dataset_dir}/steering_data.csv", "w")
    dataset_file.write("image_path,steering\n")
    
    frame_count = 0
    print(f"New dataset session created: {dataset_dir}")
    return dataset_dir

def save_frame_to_dataset(frame, steering):
    """Saves the image and steering value to the dataset"""
    global dataset_file, frame_count
    if not dataset_file:
        print("Error: No active dataset session.")
        return False
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_filename = f"frame_{timestamp}.jpg"
    image_path = f"{dataset_images_dir}/{image_filename}"
    cv2.imwrite(image_path, frame)
    
    dataset_file.write(f"images/{image_filename},{steering:.6f}\n")
    dataset_file.flush()
    
    frame_count += 1
    print(f"Frame captured: {image_path} (steering: {steering:.4f})")
    
    return True

def find_circuit_waypoints(world, vehicle):
    """Find waypoints that form a circuit in Town04"""
    map = world.get_map()
    
    # Town04 has a large highway ring road - we'll focus on finding points on this road
    # Road IDs can vary, so we'll check for a road that forms a loop
    
    # Start with vehicle's current position
    vehicle_location = vehicle.get_location()
    current_waypoint = map.get_waypoint(vehicle_location)
    
    # Try to find a larger road that might be the highway
    starting_waypoint = current_waypoint
    for attempt in range(5):
        # Look for roads with lane_id = 0 (typically main roads)
        if starting_waypoint.lane_id == 0 and starting_waypoint.lane_type == carla.LaneType.Driving:
            break
        
        # Move to a random direction to find main road
        next_options = starting_waypoint.next(50.0)
        if next_options:
            starting_waypoint = random.choice(next_options)
        else:
            # No options, try another random spawn point
            spawn_points = world.get_map().get_spawn_points()
            random_point = random.choice(spawn_points)
            starting_waypoint = map.get_waypoint(random_point.location)
    
    # Now try to find a circuit from this starting point
    print("Searching for a circuit...")
    
    # Store waypoints to form a circuit
    circuit_waypoints = []
    visited_waypoints = set()
    
    # Start with our candidate highway waypoint
    current_wp = starting_waypoint
    road_id = current_wp.road_id
    
    # Parameters to find a circuit
    circuit_search_distance = 10.0  # Distance between waypoints
    max_circuit_waypoints = 300     # Max number of waypoints to search
    
    # Start building the circuit
    for i in range(max_circuit_waypoints):
        # Add current waypoint to circuit
        circuit_waypoints.append(current_wp)
        
        # Get waypoint ID (combination of road_id, lane_id, and s-value)
        # This is used to detect when we've completed a circuit
        wp_id = (current_wp.road_id, current_wp.lane_id, round(current_wp.s, 1))
        
        # If we've seen this waypoint before and collected enough points, we may have a circuit
        if wp_id in visited_waypoints and len(circuit_waypoints) > 50:
            # Find the index where the circuit closes
            for j, wp in enumerate(circuit_waypoints):
                if (wp.road_id, wp.lane_id, round(wp.s, 1)) == wp_id and j < len(circuit_waypoints) - 10:
                    # We have a circuit from j to end
                    circuit_waypoints = circuit_waypoints[j:]
                    print(f"Circuit found with {len(circuit_waypoints)} waypoints!")
                    return circuit_waypoints
        
        # Mark this waypoint as visited
        visited_waypoints.add(wp_id)
        
        # Get next waypoint
        next_waypoints = current_wp.next(circuit_search_distance)
        
        if not next_waypoints:
            print("Failed to find a complete circuit (dead end)")
            break
        
        # Choose next waypoint (prefer staying on the same road for continuity)
        same_road_waypoints = [wp for wp in next_waypoints if wp.road_id == road_id]
        
        if same_road_waypoints:
            current_wp = same_road_waypoints[0]
        else:
            current_wp = next_waypoints[0]
    
    # If we get here and haven't found a circuit, use what we have so far
    print(f"Could not find a complete circuit. Using {len(circuit_waypoints)} waypoints as route.")
    return circuit_waypoints

def create_reverse_route(waypoints):
    """Creates a reverse route from an existing route"""
    reverse_waypoints = waypoints.copy()
    reverse_waypoints.reverse()
    
    print(f"Reverse route created with {len(reverse_waypoints)} waypoints")
    return reverse_waypoints

def process_image(image):
    """Callback function to process camera images"""
    global current_image
    # Convert to OpenCV format
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    current_image = array[:, :, :3].copy()

def drive_waypoints(world, vehicle, waypoints, speed_factor=0.5):
    """
    Drives the vehicle following the defined waypoints.
    Data capture only occurs manually when the user presses Enter.
    Returns False if user exits, True if route completed normally.
    """
    global current_image
    
    # Initial settings
    current_waypoint_index = 0
    total_waypoints = len(waypoints)
    vehicle_control = carla.VehicleControl()
    
    # Main loop
    running = True
    
    # Information about available commands
    print("\n=== AVAILABLE COMMANDS ===")
    print("ENTER - Capture frame to dataset")
    print("T - Create new dataset session")
    print("ESC - Exit program")
    print("P - Pause automatic navigation")
    print("R - Resume navigation (if paused)")
    print("+ - Increase speed")
    print("- - Decrease speed")
    print("===========================\n")
    
    paused = False
    
    while running and current_waypoint_index < total_waypoints:
        # Update the world in synchronous mode
        world.tick()
        
        # Get the next waypoint
        target_waypoint = waypoints[current_waypoint_index]
        
        if not paused:
            # Calculate vehicle and waypoint positions
            vehicle_transform = vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            target_location = target_waypoint.transform.location
            
            # Calculate direction and distance
            direction = target_location - vehicle_location
            distance = math.sqrt(direction.x**2 + direction.y**2)
            
            # If close enough, advance to the next waypoint
            if distance < 2.0:  # 2 meters tolerance
                current_waypoint_index += 1
                if current_waypoint_index % 10 == 0:
                    print(f"Waypoint {current_waypoint_index}/{total_waypoints}")
                if current_waypoint_index >= total_waypoints:
                    print("Route completed!")
                    break
                continue
            
            # Calculate angle between vehicle and waypoint
            vehicle_forward = vehicle_transform.get_forward_vector()
            dot = vehicle_forward.x * direction.x + vehicle_forward.y * direction.y
            cross = vehicle_forward.x * direction.y - vehicle_forward.y * direction.x
            angle = math.atan2(cross, dot)
            
            # Convert angle to steering value [-1, 1]
            steering = max(-1.0, min(1.0, angle * 2.0))
            
            # Configure vehicle control
            vehicle_control.throttle = speed_factor  # Speed based on factor
            vehicle_control.steer = steering
            vehicle_control.brake = 0.0
            
            # Apply control
            vehicle.apply_control(vehicle_control)
        
        # Show images
        if current_image is not None:
            img_display = current_image.copy()
            
            # Add information to the image
            velocity = vehicle.get_velocity()
            speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
            current_steering = vehicle_control.steer
            
            # Basic information
            info = f"Speed: {speed:.1f} km/h | Steering: {current_steering:.2f}"
            cv2.putText(img_display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Progress information
            progress = f"Waypoint: {current_waypoint_index}/{total_waypoints}"
            cv2.putText(img_display, progress, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Dataset information
            dataset_info = f"Dataset: {dataset_dir if dataset_file else 'No active session'}"
            cv2.putText(img_display, dataset_info, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Navigation status
            if paused:
                cv2.putText(img_display, "NAVIGATION PAUSED (press 'R' to continue)", 
                           (10, HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Speed factor
            cv2.putText(img_display, f"Speed factor: {speed_factor:.2f}", 
                       (WIDTH - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Commands
            cv2.putText(img_display, "ENTER: Capture | T: New session | +/-: Speed | ESC: Exit", 
                       (10, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('CARLA - Circuit Navigation', img_display)
            
            # Key processing
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                running = False
                print("Navigation interrupted by user.")
                return False
            elif key == 13:  # ENTER
                # Capture frame for the dataset
                if dataset_file:
                    if save_frame_to_dataset(current_image.copy(), current_steering):
                        print(f"Frame successfully captured! Total: {frame_count}")
                else:
                    print("WARNING: No active dataset session. Press 'T' to create a new session.")
            elif key == ord('t'):
                # Create new dataset session
                if dataset_file:
                    dataset_file.close()
                create_dataset_session()
                print(f"New dataset session created: {dataset_dir}")
            elif key == ord('p') or key == ord('r'):
                # Pause/resume navigation
                paused = not paused if key == ord('p') else False
                print(f"Navigation {'paused' if paused else 'resumed'}.")
                if paused:
                    # Stop the vehicle
                    vehicle_control.throttle = 0.0
                    vehicle_control.brake = 1.0
                    vehicle.apply_control(vehicle_control)
            elif key == ord('+') or key == ord('='):
                # Increase speed
                speed_factor = min(1.0, speed_factor + 0.05)
                print(f"Speed increased to: {speed_factor:.2f}")
            elif key == ord('-') or key == ord('_'):
                # Decrease speed
                speed_factor = max(0.1, speed_factor - 0.05)
                print(f"Speed decreased to: {speed_factor:.2f}")
    
    # Stop the vehicle at the end of the route
    vehicle_control.throttle = 0.0
    vehicle_control.brake = 1.0
    vehicle.apply_control(vehicle_control)
    
    print("Vehicle stopped.")
    return True  # Route completed normally

def main():
    parser = argparse.ArgumentParser(description='CARLA Circuit Navigator')
    parser.add_argument('--map', type=str, default='Town04', help='Map to use (Town04 recommended for circuits)')
    parser.add_argument('--speed', type=float, default=0.5, help='Initial speed factor (0.1-1.0)')
    args = parser.parse_args()
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    # Load the specified map (Town04 is best for circuits)
    try:
        world = client.load_world(args.map)
        print(f"Map loaded: {args.map}")
    except Exception as e:
        print(f"Error loading map: {e}")
        world = client.get_world()
        print(f"Using current map: {world.get_map().name}")
    
    # Reduce rendering complexity to improve performance
    world.unload_map_layer(carla.MapLayer.All)
    world.load_map_layer(carla.MapLayer.Ground)
    world.unload_map_layer(carla.MapLayer.Decals)
    world.unload_map_layer(carla.MapLayer.Props)
    world.unload_map_layer(carla.MapLayer.StreetLights)
    world.unload_map_layer(carla.MapLayer.Foliage)
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    world.unload_map_layer(carla.MapLayer.Particles)
    world.unload_map_layer(carla.MapLayer.Walls)
    # world.load_map_layer(carla.MapLayer.Buildings)
    # world.load_map_layer(carla.MapLayer.Props)
    # world.load_map_layer(carla.MapLayer.StreetLights)
    
    # Enable synchronous mode
    settings = world.get_settings()
    original_settings = settings
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    print("Synchronous mode enabled")
    
    actors_list = []
    
    try:
        # Select blueprint for the vehicle
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        
        # Get spawn points
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            print("No spawn points available!")
            return
        
        # Choose a spawn point
        spawn_point = random.choice(spawn_points)
        print(f"Selected spawn point: {spawn_point}")
        
        # Create the vehicle
        print("Creating vehicle...")
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actors_list.append(vehicle)
        print(f"Vehicle created: {vehicle.type_id}")
        
        # Configure camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(WIDTH))
        camera_bp.set_attribute('image_size_y', str(HEIGHT))
        camera_bp.set_attribute('fov', '90')
        
        camera_transform = carla.Transform(
            carla.Location(x=2.0, z=1.4),
            carla.Rotation(pitch=-15)
        )
        
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actors_list.append(camera)
        print("Camera created")
        
        # Register callback function for the camera
        camera.listen(process_image)
        
        # Create window for display
        cv2.namedWindow('CARLA - Circuit Navigation', cv2.WINDOW_NORMAL)
        
        # Wait a bit for the world to initialize
        for i in range(3, 0, -1):
            print(f"Starting in {i}...")
            time.sleep(1)
            world.tick()
        
        # Find circuit waypoints
        print("Finding circuit waypoints...")
        circuit_waypoints = find_circuit_waypoints(world, vehicle)
        
        if len(circuit_waypoints) < 20:
            print("Could not find a good circuit. Please try again or use a different map.")
            return
        
        # Create initial dataset session if needed
        if not dataset_file:
            create_dataset_session()
        
        # Main driving loop - keeps alternating between forward and reverse
        direction_forward = True
        current_speed = args.speed
        loop_count = 0
        running = True
        
        print("\n=== CONTINUOUS CIRCUIT NAVIGATION STARTED ===")
        print("Press ENTER to capture frames, ESC to exit")
        
        while running:
            # Select waypoints for current direction
            current_waypoints = circuit_waypoints if direction_forward else create_reverse_route(circuit_waypoints)
            
            # Drive the current route
            print(f"\n=== {'FORWARD' if direction_forward else 'REVERSE'} CIRCUIT (Loop {loop_count+1}) ===")
            
            # This function returns False if the user pressed ESC
            if not drive_waypoints(world, vehicle, current_waypoints, current_speed):
                break
            
            # Automatically toggle direction and continue
            direction_forward = not direction_forward
            loop_count += 0.5  # Half loop complete (forward or reverse)
            
            print(f"\nRoute completed! Automatically switching direction to {('FORWARD' if direction_forward else 'REVERSE')}")
            
            # Add a short pause to stabilize after route completion
            vehicle_control = carla.VehicleControl()
            vehicle_control.throttle = 0.0
            vehicle_control.brake = 1.0
            vehicle.apply_control(vehicle_control)
            time.sleep(1)  # 1 second pause
    
    finally:
        # Restore original world settings
        world.apply_settings(original_settings)
        
        # Clean up actors
        print("Cleaning up actors...")
        for actor in actors_list:
            if actor is not None and actor.is_alive:
                actor.destroy()
        
        # Close dataset file if open
        if 'dataset_file' in globals() and dataset_file:
            dataset_file.close()
            print(f"Dataset saved with {frame_count} frames")
        
        cv2.destroyAllWindows()
        print("Simulation terminated")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Execution interrupted by user")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error: {e}")
        cv2.destroyAllWindows()